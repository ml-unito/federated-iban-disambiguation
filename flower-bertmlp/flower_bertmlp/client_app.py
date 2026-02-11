"""flower-bertmlp: A Flower / PyTorch app."""

import torch
import yaml
import flwr as fl
from flwr.common import Context
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import classification_report

from flower_bertmlp.task import (
    CharacterBertForClassification, 
    set_parameters, 
    get_parameters, 
    train, 
    test, 
    load_data,
    set_system_seed,
    DEVICE
)


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def flatten_metrics(metrics_dict, prefix=""):
    
    flat = {}
    for key, value in metrics_dict.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            # Ricorsione per dizionari annidati
            nested = flatten_metrics(value, prefix=f"{full_key}_")
            flat.update(nested)
        else:
            flat[full_key] = value
    return flat


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, X_train, y_train, X_val, y_val, batch_size, eval_config, local_epochs, training_config):
        self.client_id = client_id
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.eval_config = eval_config
        self.local_epochs = local_epochs
        self.training_config = training_config
        
        # Criterion condiviso
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        
        # ####################
        # # LOCK PER FIT CONCURRENCY
        
        # from flower_bertmlp.task import acquire_fit_lock, release_fit_lock
    
        # lock_file = acquire_fit_lock(self.client_id)
        
        # print(f"Client {self.client_id}: acquisito lock")
        # ####################
        
        # try:
            self.set_parameters(parameters)
            
            fit_metrics = {}
            
            # Pre-fit evaluation
            if self.eval_config.get("pre_fit", False):
                print(f"Client {self.client_id}: pre-fit evaluation...")
                loss, metrics, p, l = test(self.model, self.X_val, self.y_val, self.batch_size, self.criterion)
                
                cr = classification_report(l, torch.stack(p).argmax(dim=1).numpy(), output_dict=True)
                print(f"Client {self.client_id} pre-fit Classification Report:\n{cr}")
                
                # Appiattisci e aggiungi prefisso
                flat_cr = flatten_metrics(cr)
                for k, v in flat_cr.items():
                    fit_metrics[f"prefit_{k}"] = v
                    
            # Training
            print(f"Client {self.client_id}: training {self.local_epochs} epochs...")
            
            # Crea optimizer e scheduler
            lr = self.training_config.get("learning_rate", 5e-6)
            weight_decay = self.training_config.get("weight_decay", 0.01)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
            num_training_steps = (len(self.X_train) // self.batch_size) * self.local_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
            
            total_loss = 0.0
            for epoch in range(self.local_epochs):
                loss, metrics = train(self.model, self.X_train, self.y_train, self.batch_size, optimizer, self.criterion, scheduler)
                total_loss += loss
            
            avg_train_loss = total_loss / self.local_epochs
            print(f"Client {self.client_id}: training done, loss={avg_train_loss:.4f}")
            fit_metrics["train_loss"] = avg_train_loss
            
            # Post-fit evaluation
            if self.eval_config.get("post_fit", False):
                print(f"Client {self.client_id}: post-fit evaluation...")
                loss, metrics, p, l = test(self.model, self.X_val, self.y_val, self.batch_size, self.criterion)
                
                cr = classification_report(l, torch.stack(p).argmax(dim=1).numpy(), output_dict=True)
                print(f"Client {self.client_id} post-fit Classification Report:\n{cr}")
                
                # Appiattisci e aggiungi prefisso
                flat_cr = flatten_metrics(cr)
                for k, v in flat_cr.items():
                    fit_metrics[f"postfit_{k}"] = v
            
            return get_parameters(self.model), len(self.X_train), fit_metrics
        # finally:
        #     release_fit_lock(lock_file)
        #     print(f"Client {self.client_id}: rilasciato lock")

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        if not self.eval_config.get("server", True):
            return 0.0, 1, {}
        
        loss, metrics, p, l = test(self.model, self.X_val, self.y_val, self.batch_size, self.criterion)
        
        cr = classification_report(l, torch.stack(p).argmax(dim=1).numpy(), output_dict=True)
        print(f"Client {self.client_id} server Classification Report:\n{cr}")
        
        # Appiattisci il classification report
        flat_cr = flatten_metrics(cr)
        
        return loss, len(self.X_val), flat_cr


def client_fn(context: Context):
    config = load_config("config/flower_exp_bertmlp.yaml")
    
    # ID client (1-based)
    client_id = int(context.node_config["partition-id"]) + 1
    print(f"Inizializzazione client {client_id}")
    
    # Fissa il seed di sistema per replicabilità (prima di creare il modello)
    set_system_seed()
    print(f"Client {client_id}: system_seed=42 applicato")
    
    # Training config
    training_config = config.get("training", {})
    
    # Carica dati (liste raw)
    X_train, y_train, X_val, y_val, batch_size = load_data(client_id, config["data"], training_config)
    
    # Crea modello (dopo aver fissato il seed)
    model = CharacterBertForClassification(num_labels=2).to(DEVICE)
    
    # Epoche locali
    local_epochs = config.get("protocol", {}).get("local_epochs", 1)
    
    client = FlowerClient(
        client_id=client_id,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        eval_config=config["eval"],
        local_epochs=local_epochs,
        training_config=training_config
    )
    
    return client.to_client()


app = fl.client.ClientApp(client_fn=client_fn)
