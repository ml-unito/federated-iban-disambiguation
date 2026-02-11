"""flower-bertmlp: A Flower / PyTorch app."""

import os
import yaml
import flwr as fl
import wandb
import torch
from datetime import datetime
from flwr.common import Context, ndarrays_to_parameters

from flower_bertmlp.task import CharacterBertForClassification, get_parameters, weighted_average, set_system_seed, DEVICE


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def aggregate_metrics_by_prefix(metrics_list, prefix):
    
    if not metrics_list:
        return {}
    
    totals = {}
    weights = {}
    
    for num_examples, metrics_dict in metrics_list:
        for key, value in metrics_dict.items():
            if key.startswith(prefix):
                # Rimuovi il prefisso per il nome della metrica
                metric_name = key[len(prefix):]
                try:
                    val = float(value)
                    if metric_name not in totals:
                        totals[metric_name] = 0.0
                        weights[metric_name] = 0
                    totals[metric_name] += val * num_examples
                    weights[metric_name] += num_examples
                except (TypeError, ValueError):
                    continue
    
    return {k: totals[k] / weights[k] for k in totals if weights[k] > 0}


class FedAvgWithLogging(fl.server.strategy.FedAvg):
    def __init__(self, eval_config, n_rounds, output_path, **kwargs):
        super().__init__(**kwargs)
        self.eval_config = eval_config
        self.n_rounds = n_rounds
        self.output_path = output_path

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        
        if wandb.run is not None and results:
            # Estrai tutte le metriche dai risultati
            metrics_list = [(res.num_examples, res.metrics) for _, res in results]
            
            log_data = {}
            
            # Aggrega metriche pre-fit
            if self.eval_config.get("pre_fit", False):
                prefit_metrics = aggregate_metrics_by_prefix(metrics_list, "prefit_")
                for k, v in prefit_metrics.items():
                    log_data[f"prefit.{k}"] = v
            
            # Aggrega metriche post-fit
            if self.eval_config.get("post_fit", False):
                postfit_metrics = aggregate_metrics_by_prefix(metrics_list, "postfit_")
                for k, v in postfit_metrics.items():
                    log_data[f"postfit.{k}"] = v
            
            # Training loss
            train_losses = [m.get("train_loss", 0) for _, m in metrics_list if "train_loss" in m]
            if train_losses:
                log_data["train.loss"] = sum(train_losses) / len(train_losses)
            
            if log_data:
                wandb.log(log_data, step=server_round)
                print(f"Round {server_round} - Logged: {list(log_data.keys())}")
        
        # Salva modello all'ultimo round
        if params is not None and server_round == self.n_rounds:
            model = CharacterBertForClassification(num_labels=2)
            weights = fl.common.parameters_to_ndarrays(params)
            state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(p) for p in weights]))
            model.load_state_dict(state_dict)
            
            save_path = os.path.join(self.output_path, f"global_model_R{server_round}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Modello salvato: {save_path}")
        
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            print(f"Round {server_round}: nessun risultato di valutazione")
            return 0.0, {}
            
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated is None:
            return 0.0, {}
        
        loss, metrics = aggregated
        
        # Log metriche globali (server evaluation)
        if wandb.run is not None and self.eval_config.get("server", True):
            log_data = {"global.loss": loss}
            for k, v in metrics.items():
                if k not in ("client_id",):
                    log_data[f"global.{k}"] = v
            wandb.log(log_data, step=server_round)
        
        print(f"Round {server_round}: global loss={loss:.4f}, metrics={metrics}")
        return aggregated


def init_wandb(config, output_path):
    logger_config = config["logger"]
    
    wandb.init(
        project=logger_config["project"],
        entity=logger_config["entity"],
        group=logger_config["group"],
        tags=logger_config["tags"],
        config=config,
        reinit=True,
    )
    
    seed = config['data']['dataset']['seed']
    n_clients = config['protocol']['n_clients']
    n_rounds = config['protocol']['n_rounds']
    wandb.run.name = f"bert_mlp_S{seed}_{n_clients}CL_{n_rounds}R"


def server_fn(context: Context):
    config = load_config("config/flower_exp_bertmlp.yaml")
    
    # Fissa il seed di sistema per replicabilità (prima di creare il modello)
    set_system_seed()
    print("Server: system_seed=42 applicato")
    
    n_clients = config["protocol"]["n_clients"]
    n_rounds = config["protocol"]["n_rounds"]
    
    print(f"Server: {n_clients} clients, {n_rounds} rounds")
    
    # Directory output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = config["data"]["dataset"]["seed"]
    output_path = f"./out/flwr_S{seed}_{timestamp}/"
    os.makedirs(output_path, exist_ok=True)
    
    # WandB
    if config["logger"]["enabled"]:
        init_wandb(config, output_path)
    
    # Modello iniziale
    model = CharacterBertForClassification(num_labels=2)
    initial_params = ndarrays_to_parameters(get_parameters(model))
    
    # Strategia
    strategy = FedAvgWithLogging(
        eval_config=config["eval"],
        n_rounds=n_rounds,
        output_path=output_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    
    server_config = fl.server.ServerConfig(
        num_rounds=n_rounds,
        round_timeout=7200 # 2 hours timeout per round
    )
    
    return fl.server.ServerAppComponents(strategy=strategy, config=server_config)


app = fl.server.ServerApp(server_fn=server_fn)
