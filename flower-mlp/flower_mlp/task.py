"""flower-mlp: A Flower / PyTorch app."""

import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# # Importa il modello MLP dal file lib/mlp.py
# sys.path.insert(0, os.path.abspath('./lib'))
# from lib.mlp import MLP


# Definisci il device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")


class MLP(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Classe per gestire il dataset
class KernelDataset:
    def __init__(self, csv_path:str, is_train:bool):
        self.data = pd.read_csv(csv_path)
        self.is_train = is_train
        
    def get_features_and_labels(self):
        # Prendi tutte le colonne tranne l'ultima come features
        features = self.data.iloc[:, :-1].values
        # Prendi l'ultima colonna come label
        labels = self.data.iloc[:, -1].values
        
        # Converti in tensori PyTorch
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return features_tensor, labels_tensor

    def scale_features(self, scaler:MinMaxScaler):
        if self.is_train:
            self.data.iloc[:, :-1] = scaler.fit_transform(self.data.iloc[:, :-1])
        else:
            self.data.iloc[:, :-1] = scaler.transform(self.data.iloc[:, :-1])


fds = None  # Cache FederatedDataset


def load_data(client_id, config_data):
    """Carica i dati del client specificato."""
    train_path = config_data["dataset"]["sim_train_path"] % (client_id, config_data["dataset"]["seed"], "")
    test_path = config_data["dataset"]["sim_test_path"] % (config_data["dataset"]["seed"], "")
    scaler = MinMaxScaler()

    # Carica i dati di training
    train_dataset = KernelDataset(train_path, is_train=True)
    train_dataset.scale_features(scaler)
    train_features, train_labels = train_dataset.get_features_and_labels()

    train_tensor_dataset = TensorDataset(train_features, train_labels)
    trainloader = DataLoader(train_tensor_dataset, batch_size=1024, shuffle=True)
    
    # Carica i dati di test
    test_dataset = KernelDataset(test_path, is_train=False)
    test_dataset.scale_features(scaler)
    test_features, test_labels = test_dataset.get_features_and_labels()

    test_tensor_dataset = TensorDataset(test_features, test_labels)
    testloader = DataLoader(test_tensor_dataset, batch_size=1024)
    
    return trainloader, testloader


def set_parameters(model, parameters):
    """Set model parameters from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model):
    """Get model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def train(model, trainloader, epochs=4, device=DEVICE):
    """Train the model on local data."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    running_loss = 0.0
    print(f"[DEBUG] Inizio training su {device} ({len(trainloader)} batch)")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for features, labels in trainloader:
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count}/{len(trainloader)}")
            batch_count += 1
            
            try:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            except Exception as e:
                print(f"[ERROR] Errore durante il training: {e}")
                raise e
        
        running_loss += epoch_loss
        print(f"Epoch {epoch+1}/{epochs}: loss {epoch_loss/len(trainloader)}")
    
    print("[DEBUG] Training completato con successo!")
    return running_loss / (epochs * len(trainloader))


def test(model, testloader, device=DEVICE):
    """Evaluate the model on the test dataset."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    loss = 0.0
    correct, total = 0, 0
    
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = loss / len(testloader) if len(testloader) > 0 else 0
    
    return avg_loss, accuracy


def log_detailed_metrics(model, train_x, train_y, test_x, test_y, loss=None, round_num=0, phase="global"):
    """Log metriche dettagliate su WandB."""
    
    with torch.no_grad():
        train_preds = model(train_x.to(DEVICE)).argmax(dim=1).cpu().numpy()
        test_preds = model(test_x.to(DEVICE)).argmax(dim=1).cpu().numpy()
        
        # Converti a numpy per la valutazione
        train_y_np = train_y.cpu().numpy()
        test_y_np = test_y.cpu().numpy()
        
        # Report di classificazione
        cr_train = classification_report(train_y_np, train_preds, output_dict=True, zero_division=0)
        cr_test = classification_report(test_y_np, test_preds, output_dict=True, zero_division=0)
        
        # Crea le metriche, ora anche con micro_f1
        metrics = {
            "train_accuracy": float(cr_train["accuracy"]),
            "test_accuracy": float(cr_test["accuracy"]),
            "train_macro_f1": float(cr_train["macro avg"]["f1-score"]),
            "test_macro_f1": float(cr_test["macro avg"]["f1-score"]),
            "train_micro_f1": float(cr_train["weighted avg"]["f1-score"]),  # utilizziamo weighted avg come micro_f1
            "test_micro_f1": float(cr_test["weighted avg"]["f1-score"]),
            "train_precision": float(cr_train["macro avg"]["precision"]),
            "test_precision": float(cr_test["macro avg"]["precision"]),
            "train_recall": float(cr_train["macro avg"]["recall"]),
            "test_recall": float(cr_test["macro avg"]["recall"]),
        }
        
        # Aggiungi loss se disponibile
        if loss is not None:
            metrics["train_loss"] = float(loss)
            
        # Logga su WandB con il contesto della fase - solo global
        if wandb.run is not None:
            phase_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
            phase_metrics["round"] = round_num
            wandb.log(phase_metrics)
            
        return metrics


def weighted_average(metrics):
    """Aggregazione ponderata delle metriche."""
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
                # Converte value a float se possibile
                value = float(value) if isinstance(value, str) else value
                
                # Calcola la somma ponderata
                metrics_sum[key] += value * num_examples
                examples_sum[key] += num_examples
            except (TypeError, ValueError):
                # Se c'è un errore di tipo, salta questo valore
                continue
    
    # Calcola la media ponderata per ogni metrica
    weighted_metrics = {}
    for key in metrics_sum:
        if examples_sum[key] > 0:  # Evita la divisione per zero
            weighted_metrics[key] = metrics_sum[key] / examples_sum[key]
    
    return weighted_metrics
