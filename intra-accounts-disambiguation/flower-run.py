import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import wandb
from typing import Dict, List, Tuple
import random
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler

# Importa il modello MLP dal file lib/mlp.py
sys.path.insert(0, os.path.abspath('./lib'))
from lib.mlp import MLP

# Impostazione seed per riproducibilità
# SEED = 47874
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# Definisci il device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

# Classe per gestire il dataset
class KernelDataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.scaler = MinMaxScaler()
        
    def get_features_and_labels(self):
        # Prendi tutte le colonne tranne l'ultima come features
        features = self.data.iloc[:, :-1].values
        # Normalizza le features
        features = self.scaler.fit_transform(features)
        # Prendi l'ultima colonna come label
        labels = self.data.iloc[:, -1].values
        
        # Converti in tensori PyTorch
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return features_tensor, labels_tensor

# Funzioni helper per gestire i parametri del modello
def set_parameters(model, parameters: List[np.ndarray]):
    """Set model parameters from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_parameters(model) -> List[np.ndarray]:
    """Get model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Utilità per calcolare le metriche
def compute_metrics(predictions, labels):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions, zero_division=0)
    precision = precision_score(y_true=labels, y_pred=predictions, zero_division=0)
    f1 = f1_score(y_true=labels, y_pred=predictions, zero_division=0)
    
    return {
        "accuracy": round(accuracy, 3), 
        "precision": round(precision, 3), 
        "recall": round(recall, 3), 
        "f1": round(f1, 3)
    }

# Funzioni di training e valutazione
def train(model, trainloader, epochs=1, verbose=False):
    """Train the model on local data."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct, total = 0, 0
        
        for features, labels in trainloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        if verbose:
            print(f"Epoch {epoch+1}: train loss {total_loss/len(trainloader)}, accuracy {correct/total}")

def test(model, testloader):
    """Evaluate the model on the test dataset."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    loss = 0.0
    correct, total = 0, 0
    
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(features)
            loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return loss / len(testloader), correct / total

# Funzione di aggregazione delle metriche
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregazione ponderata delle metriche."""
    # Estrai i valori per ogni metrica
    metrics_sum = {}
    examples_sum = {}
    
    for num_examples, metrics_dict in metrics:
        for key, value in metrics_dict.items():
            # Inizializza se la chiave non esiste
            if key not in metrics_sum:
                metrics_sum[key] = 0
                examples_sum[key] = 0
                
            # Verifica che il valore sia numerico prima di moltiplicarlo
            try:
                # Converte value a float se è una stringa che rappresenta un numero
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        # Se non è convertibile, salta questo valore
                        continue
                
                # Ora possiamo calcolare la somma ponderata
                metrics_sum[key] += value * num_examples
                examples_sum[key] += num_examples
            except TypeError:
                # Se c'è un errore di tipo, salta questo valore
                continue
    
    # Calcola la media ponderata per ogni metrica
    weighted_metrics = {}
    for key in metrics_sum:
        if examples_sum[key] > 0:  # Evita la divisione per zero
            weighted_metrics[key] = metrics_sum[key] / examples_sum[key]
    
    return weighted_metrics

# Definizione del client Flower con l'API NumPyClient
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, eval_config):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.eval_config = eval_config  # Salva la configurazione eval
    
    def get_parameters(self, config):
        return get_parameters(self.model)
    
    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)
    
    def fit(self, parameters, config):
        # Ottieni round corrente per logging
        round_num = config.get("round_num", 0)
        
        # Prepara i dati per la valutazione
        all_train_features = []
        all_train_labels = []
        all_test_features = []
        all_test_labels = []
        
        for features, labels in self.trainloader:
            all_train_features.append(features)
            all_train_labels.append(labels)
        for features, labels in self.valloader:
            all_test_features.append(features)
            all_test_labels.append(labels)
        
        train_x = torch.cat(all_train_features, dim=0)
        train_y = torch.cat(all_train_labels, dim=0)
        test_x = torch.cat(all_test_features, dim=0)
        test_y = torch.cat(all_test_labels, dim=0)
        
        # Valutazione PRE-FIT solo se configurato
        self.set_parameters(parameters)
        self.model.eval()
        
        # Log prefit solo se abilitato nella configurazione
        if self.eval_config["pre_fit"]:
            log_detailed_metrics(
                self.model, 
                train_x, train_y, 
                test_x, test_y,
                None,  # No loss disponibile prima del training
                round_num,
                phase="prefit"
            )
        
        # TRAINING
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        running_loss = 0.0
        
        for epoch in range(4):  # 4 epoche locali come specificato nei tag
            epoch_loss = 0.0
            for features, labels in self.trainloader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            running_loss += epoch_loss
        
        # Valutazione POST-FIT solo se configurato
        self.model.eval()
        
        # Log postfit solo se abilitato nella configurazione
        if self.eval_config["post_fit"]:
            log_detailed_metrics(
                self.model, 
                train_x, train_y, 
                test_x, test_y, 
                running_loss / (4 * len(self.trainloader)),
                round_num,
                phase="postfit"
            )
        
        # Return updated parameters and training stats
        return get_parameters(self.model), len(self.trainloader.dataset), {"loss": running_loss / 4}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        round_num = config.get("round_num", 0)
        
        # Non procedere con la valutazione se server=false
        if not self.eval_config["server"]:
            return 0.0, 0, {}
        
        # Prepara i dati di test
        all_test_features = []
        all_test_labels = []
        
        for features, labels in self.valloader:
            all_test_features.append(features)
            all_test_labels.append(labels)
            
        test_x = torch.cat(all_test_features, dim=0)
        test_y = torch.cat(all_test_labels, dim=0)
        
        # Prepara i dati di training
        all_train_features = []
        all_train_labels = []
        
        for features, labels in self.trainloader:
            all_train_features.append(features)
            all_train_labels.append(labels)
            
        train_x = torch.cat(all_train_features, dim=0)
        train_y = torch.cat(all_train_labels, dim=0)
        
        # Valuta il modello
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            outputs = self.model(test_x.to(DEVICE))
            loss = criterion(outputs, test_y.to(DEVICE)).item()
        
        # Log delle metriche dettagliate per la fase "global"
        metrics = log_detailed_metrics(
            self.model, 
            train_x, train_y, 
            test_x, test_y, 
            loss,
            round_num,
            phase="global"
        )
        
        return loss, len(test_y), metrics

# Aggiungi questa funzione per il logging dettagliato di metriche
def log_detailed_metrics(model, train_x, train_y, test_x, test_y, loss=None, round_num=0, phase="global"):
    """Log metriche dettagliate su WandB, come in kernel-classify.py ma senza divisione per etichetta"""
    from sklearn.metrics import classification_report
    
    with torch.no_grad():
        train_preds = model(train_x.to(DEVICE)).argmax(dim=1).cpu().numpy()
        test_preds = model(test_x.to(DEVICE)).argmax(dim=1).cpu().numpy()
        
        # Converti a numpy per la valutazione
        train_y_np = train_y.cpu().numpy()
        test_y_np = test_y.cpu().numpy()
        
        # Report di classificazione
        cr_train = classification_report(train_y_np, train_preds, output_dict=True, zero_division=0)
        cr_test = classification_report(test_y_np, test_preds, output_dict=True, zero_division=0)
        
        # Crea tre set di metriche separati ma con gli stessi nomi base
        # Questo permette a WandB di raggrupparli automaticamente nei grafici
        metrics = {
            "train_accuracy": cr_train["accuracy"],
            "test_accuracy": cr_test["accuracy"],
            "train_macro_f1": cr_train["macro avg"]["f1-score"],
            "test_macro_f1": cr_test["macro avg"]["f1-score"],
            "train_precision": cr_train["macro avg"]["precision"],
            "test_precision": cr_test["macro avg"]["precision"],
            "train_recall": cr_train["macro avg"]["recall"],
            "test_recall": cr_test["macro avg"]["recall"],
        }
        
        # Aggiungi loss se disponibile
        if loss is not None:
            metrics["train_loss"] = loss
            
        # Logga su WandB con il contesto della fase
        if wandb.run is not None:
            # Crea un gruppo di metriche prefissate con la fase
            phase_metrics = {f"{phase}/{k}": v for k, v in metrics.items() if k not in ["phase", "round"]}
            phase_metrics["round"] = round_num
            wandb.log(phase_metrics)
            
        return metrics

# Funzione per caricare la configurazione
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Funzione per inizializzare WandB
def init_wandb(config):
    logger_config = config["logger"]
    wandb.init(
        project=logger_config["project"],
        entity=logger_config["entity"],
        group=logger_config["group"],
        tags=logger_config["tags"] + ["prefit", "postfit", "global"],  # Aggiungi tag per le fasi
        config=config,
        reinit=True
    )
    
    # Definisci un layout personalizzato per WandB
    wandb.run.log_code(".")  # Salva il codice sorgente
    
    # Configura la visualizzazione per le metriche principali
    for metric in ["accuracy", "macro_f1", "precision", "recall"]:
        for dataset in ["train", "test"]:
            wandb.define_metric(f"prefit/{dataset}_{metric}", step_metric="round")
            wandb.define_metric(f"postfit/{dataset}_{metric}", step_metric="round")
            wandb.define_metric(f"global/{dataset}_{metric}", step_metric="round")

# Funzione di creazione del client che riceve un Context
def client_fn(context: fl.common.Context) -> fl.client.Client:
    # Carica la configurazione
    config = load_config("config/flower_exp_kernel_nn.yaml")  # Usa flower_exp_kernel_nn.yaml
    
    # Ottieni l'ID del client dalla configurazione del nodo
    client_id = int(context.node_config["partition-id"]) + 1
    
    # Carica i dati del client
    train_path = config["data"]["dataset"]["sim_train_path"] % (client_id, config["data"]["dataset"]["seed"], "")
    dataset = KernelDataset(train_path)
    features, labels = dataset.get_features_and_labels()
    
    # Crea un DataLoader per il training
    train_dataset = TensorDataset(features, labels)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Carica il test set se necessario
    test_path = config["data"]["dataset"]["sim_test_path"] % (config["data"]["dataset"]["seed"], "")
    test_dataset = KernelDataset(test_path)
    test_features, test_labels = test_dataset.get_features_and_labels()
    val_dataset = TensorDataset(test_features, test_labels)
    valloader = DataLoader(val_dataset, batch_size=32)
    
    # Crea il modello
    input_size = features.shape[1]
    model = MLP(input_dim=input_size, hidden_dim=128, output_dim=2).to(DEVICE)
    
    # Crea e restituisci il client Flower con le impostazioni eval
    return FlowerClient(
        model=model,
        trainloader=trainloader, 
        valloader=valloader,
        eval_config=config["eval"]  # Passa la configurazione eval
    ).to_client()

# Funzione di creazione del server che riceve un Context
def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
    # Carica la configurazione
    config = load_config("config/flower_exp_kernel_nn.yaml")  # Usa flower_exp_kernel_nn.yaml
    
    # Crea la strategia FedAvg con logging
    strategy = FedAvgWithLogging(
        eval_config=config["eval"],  # Passa la configurazione eval
        fraction_fit=config["protocol"]["eligible_perc"],
        fraction_evaluate=config["protocol"]["eligible_perc"],
        min_fit_clients=config["protocol"]["n_clients"],
        min_evaluate_clients=config["protocol"]["n_clients"],
        min_available_clients=config["protocol"]["n_clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average
    )
    
    # Configura il server
    server_config = fl.server.ServerConfig(num_rounds=config["protocol"]["n_rounds"])
    
    # Crea e restituisci i componenti del server (senza callback)
    return fl.server.ServerAppComponents(
        strategy=strategy, 
        config=server_config
    )

# Nuova strategia FedAvg with logging
class FedAvgWithLogging(fl.server.strategy.FedAvg):
    def __init__(self, eval_config, **kwargs):
        super().__init__(**kwargs)
        self.eval_config = eval_config
        
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        # Log delle metriche aggregate solo se server=true nella configurazione
        if aggregated_result is not None and wandb.run is not None and self.eval_config["server"]:
            loss, metrics = aggregated_result
            
            # Usa lo stesso schema di naming per la fase global
            global_metrics = {
                "global/train_loss": loss,
                "round": server_round
            }
            
            # Aggiungi tutte le metriche con il prefisso "global/"
            for k, v in metrics.items():
                global_metrics[f"global/{k}"] = v
            
            # Log su WandB
            wandb.log(global_metrics)
            print(f"Round {server_round} completato: {metrics}")
        
        return aggregated_result

# Funzione principale
def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning con MLP")
    parser.add_argument("--config", type=str, default="config/flower_exp_kernel_nn.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Carica la configurazione
    config = load_config(args.config)
    
    # Inizializza WandB
    init_wandb(config)
    
    # Crea le applicazioni server e client
    server_app = fl.server.ServerApp(server_fn=server_fn)
    client_app = fl.client.ClientApp(client_fn=client_fn)
    
    # Configura le risorse per ogni client
    resources = {"num_cpus": 1}
    if torch.cuda.is_available():
        resources["num_gpus"] = 0.1  # Usa una piccola porzione di GPU per ogni client
    
    # Avvia la simulazione
    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config["protocol"]["n_clients"],
        backend_config={"client_resources": resources}
    )
    
    # Chiudi WandB
    wandb.finish()

if __name__ == "__main__":
    main()