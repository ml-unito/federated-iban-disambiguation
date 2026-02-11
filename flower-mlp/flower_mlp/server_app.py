"""flower-mlp: A Flower / PyTorch app."""

import yaml
import flwr as fl
import wandb
import os
import torch
from datetime import datetime
from flwr.common import Context, ndarrays_to_parameters

from flower_mlp.task import MLP, get_parameters, weighted_average, DEVICE, set_system_seed


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


class FedAvgWithLogging(fl.server.strategy.FedAvg):
    def __init__(self, eval_config, rounds, path, **kwargs):
        super().__init__(**kwargs)
        self.eval_config = eval_config
        self.rounds = rounds
        self.path = path

    def aggregate_evaluate(self, server_round, results, failures):
        print(f"[DEBUG] Aggregate_evaluate: round={server_round}, results={len(results)}, failures={len(failures)}")
        
        # Gestisci il caso di nessun risultato
        if not results:
            print(f"[WARN] Round {server_round}: Nessun risultato di valutazione ricevuto dai client.")
            # Restituisci un risultato fittizio per evitare None
            return 0.0, {"accuracy": 0.0, "loss": 0.0, "train_macro_f1": 0.0, "test_macro_f1": 0.0}
            
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        # Se anche dopo l'aggregazione non ci sono risultati, crea un risultato vuoto
        if aggregated_result is None:
            print(f"[WARN] Round {server_round}: Aggregazione risultati fallita.")
            return 0.0, {"accuracy": 0.0, "loss": 0.0, "train_macro_f1": 0.0, "test_macro_f1": 0.0}
        
        # Log delle metriche aggregate
        loss, metrics = aggregated_result

        # Log delle metriche aggregate solo se server=true nella configurazione
        if (
            aggregated_result is not None
            and wandb.run is not None
            and self.eval_config["server"]
        ):
            # Usa lo stesso schema di naming per la fase global
            global_metrics = {"train_loss": loss, "round": server_round}

            # Aggiungi tutte le metriche con il prefisso "global/"
            for k, v in metrics.items():
                global_metrics[f"{k}"] = v

            # Log su WandB
            wandb.log(global_metrics)
            print(f"Round {server_round} completato: {metrics}")

        return aggregated_result
    
    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if parameters_aggregated is not None and server_round == self.rounds:
            # Saving the global model at the end of all rounds.
            model = MLP(input_dim=7, hidden_dim=128, output_dim=2)
            params = fl.common.parameters_to_ndarrays(parameters_aggregated)
            state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(p) for p in params]))
            model.load_state_dict(state_dict)
            torch.save(model.state_dict(),self.path + "global_model_R" + str(server_round) + ".pt")
        
        return parameters_aggregated, metrics_aggregated


def init_wandb(config: dict, path_model_dir: str):
    logger_config = config["logger"]
    config["output_dir"] = path_model_dir
    
    wandb.init(
        project=logger_config["project"],
        entity=logger_config["entity"],
        group=logger_config["group"],
        tags=logger_config["tags"] + ["global"],  # Solo tag global
        config=config,
        reinit=True,
    )
    
    wandb.run.name = f"flower_kernel_mlp_{config['data']['dataset']['seed']}_{config['protocol']['n_clients']}CL_{config['protocol']['n_rounds']}R"

    # Definisci un layout personalizzato per WandB
    wandb.run.log_code(".")

    # Configura la visualizzazione per le metriche principali - solo global
    for metric in ["accuracy", "macro_f1", "micro_f1", "precision", "recall"]:  # Aggiunto micro_f1
        for dataset in ["train", "test"]:
            wandb.define_metric(f"global.{dataset}_{metric}", step_metric="round")


def server_fn(context: Context):
    
    # Carica la configurazione
    config = load_config("config/flower_exp_kernel_nn.yaml")
    
    # Fissa il seed di sistema per replicabilità (prima di creare il modello)
    set_system_seed()
    print("Server: system_seed=42 applicato")
    
    # Debug per verificare i valori di configurazione compreso il seeding
    print(f"[DEBUG] Configurazione caricata: {config}")
    print(f"[DEBUG] Seeding: {config['data']['dataset']['seed']}")
    print(f"[DEBUG] Configurazione server: n_clients={config['protocol']['n_clients']}, "
          f"rounds={config['protocol']['n_rounds']}")
    
    
    # Assicurati che l'eval del server sia attivato
    if not config["eval"]["server"]:
        print("ATTENZIONE: La valutazione sul server è disabilitata (eval.server=false).")
        print("Impostazione forzata a True per evitare errori.")
        config["eval"]["server"] = True

    date = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-") 
    path_model_dir = "./out/flwr_S"+str(config["data"]["dataset"]["seed"])+"_"+date+"/"
    os.makedirs(path_model_dir)
    
    # Inizializza WandB se richiesto
    if config.get("use_wandb", True):
        init_wandb(config, path_model_dir)

    # Parametri iniziali
    dummy_model = MLP(input_dim=7, hidden_dim=128, output_dim=2)
    initial_parameters = ndarrays_to_parameters(get_parameters(dummy_model))

    # Crea la strategia FedAvg con logging
    strategy = FedAvgWithLogging(
        eval_config=config["eval"],
        rounds=config["protocol"]["n_rounds"],
        path=path_model_dir,
        fraction_fit=1.0,  # Imposta a 1.0 per utilizzare tutti i client disponibili
        fraction_evaluate=1.0,
        min_fit_clients=4,  # Imposta a 1 per iniziare anche con un solo client
        min_evaluate_clients=4,
        min_available_clients=4,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    # Configura il server con timeout
    server_config = fl.server.ServerConfig(
        num_rounds=config["protocol"]["n_rounds"],
        round_timeout=600  # 10 minuti di timeout per round
    )

    # Crea e restituisci i componenti del server
    return fl.server.ServerAppComponents(strategy=strategy, config=server_config)


# Flower ServerApp
app = fl.server.ServerApp(server_fn=server_fn)
