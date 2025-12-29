"""flower-bertmlp: A Flower / PyTorch app."""

import torch
import yaml
import flwr as fl
from flwr.common import Context

from flower_bertmlp.task import (
    CharacterBertForClassification, 
    set_parameters, 
    get_parameters, 
    train, 
    test, 
    load_data, 
    DEVICE
)


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, trainloader, valloader, eval_config, local_epochs, training_config):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.eval_config = eval_config
        self.local_epochs = local_epochs
        self.training_config = training_config

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        fit_metrics = {}
        
        # Pre-fit evaluation
        if self.eval_config.get("pre_fit", False):
            print(f"Client {self.client_id}: pre-fit evaluation...")
            loss, accuracy, metrics = test(self.model, self.valloader, device=DEVICE)
            print(f"Client {self.client_id}: pre-fit loss={loss:.4f}, acc={accuracy:.4f}")
            
            # Aggiungi metriche con prefisso prefit
            for k, v in metrics.items():
                fit_metrics[f"prefit_{k}"] = v
        
        # Training
        print(f"Client {self.client_id}: training {self.local_epochs} epochs...")
        train_loss = train(
            self.model, 
            self.trainloader, 
            epochs=self.local_epochs, 
            device=DEVICE,
            lr=self.training_config.get("learning_rate", 5e-6),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0)
        )
        print(f"Client {self.client_id}: training done, loss={train_loss:.4f}")
        fit_metrics["train_loss"] = train_loss
        
        # Post-fit evaluation
        if self.eval_config.get("post_fit", False):
            print(f"Client {self.client_id}: post-fit evaluation...")
            loss, accuracy, metrics = test(self.model, self.valloader, device=DEVICE)
            print(f"Client {self.client_id}: post-fit loss={loss:.4f}, acc={accuracy:.4f}")
            
            # Aggiungi metriche con prefisso postfit
            for k, v in metrics.items():
                fit_metrics[f"postfit_{k}"] = v
        
        return get_parameters(self.model), len(self.trainloader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        """Valutazione chiamata dal server per aggregare metriche globali."""
        self.set_parameters(parameters)
        
        # Questa evaluate viene usata per la valutazione "server" aggregata
        if not self.eval_config.get("server", True):
            return 0.0, 1, {}
        
        loss, accuracy, metrics = test(self.model, self.valloader, device=DEVICE)
        print(f"Client {self.client_id}: server eval loss={loss:.4f}, acc={accuracy:.4f}")
        
        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    config = load_config("config/flower_exp_bertmlp.yaml")
    
    # ID client (1-based)
    client_id = int(context.node_config["partition-id"]) + 1
    print(f"Inizializzazione client {client_id}")
    
    # Training config
    training_config = config.get("training", {})
    
    # Carica dati (senza raw_data)
    trainloader, valloader = load_data(client_id, config["data"], training_config)
    
    # Crea modello
    model = CharacterBertForClassification(num_labels=2).to(DEVICE)
    
    # Epoche locali
    local_epochs = config.get("protocol", {}).get("local_epochs", 1)
    
    client = FlowerClient(
        client_id=client_id,
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        eval_config=config["eval"],
        local_epochs=local_epochs,
        training_config=training_config
    )
    
    return client.to_client()


app = fl.client.ClientApp(client_fn=client_fn)
