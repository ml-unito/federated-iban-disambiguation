"""flower-bertmlp: A Flower / PyTorch app."""

import os
import sys
import json
from collections import OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import classification_report
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig

from lib.datasetManipulation import *

# Importa il modello CharacterBERT
sys.path.insert(0, os.path.abspath('./character_bert_model'))
from character_bert_model.utils.character_cnn import CharacterIndexer
from character_bert_model.modeling.character_bert import CharacterBertModel

# Importa e esegui il download del modello
from flower_bertmlp.download import download_pre_trained_model
download_pre_trained_model()

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

# Indexer per tokenizzazione
indexer = CharacterIndexer()


def tokenize_pair(text):
    """Tokenizza una coppia di testo separata da '@'"""
    parts = text.split("@")
    if len(parts) == 2:
        tokens = ['[CLS]'] + parts[0].strip().split() + ['[SEP]'] + parts[1].strip().split() + ['[SEP]']
    else:
        tokens = ['[CLS]'] + text.strip().split() + ['[SEP]']
    return tokens


class CharacterBertForClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(CharacterBertForClassification, self).__init__()
        
        model_path = './character_bert_model/pretrained-models/general_character_bert/'
        
        try:
            self.character_bert = CharacterBertModel.from_pretrained(model_path)
        except Exception as e:
            print(f"Caricamento alternativo del modello: {e}")
            config_path = os.path.join(model_path, 'config.json')
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = BertConfig(**config_dict)
            self.character_bert = CharacterBertModel(config)
            
            weights_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
                self.character_bert.load_state_dict(state_dict, strict=False)
        
        self.dropout = nn.Dropout(0.1)  # Ridotto da 0.2
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids):
        outputs = self.character_bert(input_ids)[0]
        pooled_output = self.dropout(outputs[:, 0, :])
        logits = self.classifier(pooled_output)
        return logits


def load_csv_pairs(csv_path: str, balance: bool = False) -> tuple:
    """Carica coppie di testo e labels da un CSV preprocessato."""
    data = pd.read_csv(csv_path)
    print(f"Caricato {csv_path}: {len(data)} righe, colonne: {data.columns.tolist()}")
    
    pairs = []
    labels = []
    
    # Formato già pronto con text e label
    if 'text' in data.columns and 'label' in data.columns:
        if balance:
            data = balance_dataset(data, 'label', oversample=True)
        pairs = data['text'].tolist()
        labels = data['label'].astype(int).tolist()
    
    # Formato raw con Name, cluster e AccountNumber - crea coppie
    elif 'Name' in data.columns and 'AccountNumber' in data.columns:
        print(f"  Creazione coppie da {len(data)} righe...")
        couple_df = create_pairs(data)
        print(f"  -> Coppie create: {len(couple_df)}")
        
        if balance:
            print(f"  Bilanciamento dataset...")
            couple_df = balance_dataset(couple_df, 'label', oversample=True)
            print(f"  -> Coppie dopo bilanciamento: {len(couple_df)}")
        
        # create_pairs restituisce DataFrame con colonne 'text' e 'label'
        pairs = couple_df['text'].tolist()
        labels = couple_df['label'].astype(int).tolist()
    
    else:
        raise ValueError(f"Formato CSV non riconosciuto. Colonne: {data.columns.tolist()}")
    
    print(f"  -> {len(pairs)} coppie finali, distribuzione: 0={labels.count(0)}, 1={labels.count(1)}")
    return pairs, labels


def create_dataloader(texts, labels, batch_size=32, shuffle=True):
    """Crea un DataLoader da testi e labels."""
    print(f"  Tokenizzazione di {len(texts)} testi...")
    tokenized = [tokenize_pair(text) for text in texts]
    
    print(f"  Creazione tensori...")
    input_ids = indexer.as_padded_tensor(tokenized)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    print(f"  Shape input_ids: {input_ids.shape}, labels: {labels_tensor.shape}")
    
    dataset = TensorDataset(input_ids, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_data(client_id, config_data, training_config=None):
    """Carica i dati del client specificato."""
    dataset_config = config_data["dataset"]
    
    # Determina il path di training per questo client
    if "client_train_paths" in dataset_config:
        client_train_paths = dataset_config["client_train_paths"]
        idx = client_id - 1
        if idx < len(client_train_paths):
            train_path = client_train_paths[idx]
        else:
            raise ValueError(f"Client ID {client_id} fuori range (max: {len(client_train_paths)})")
    else:
        train_path = dataset_config["train_path"]
    
    test_path = dataset_config["test_path"]
    
    # Opzione di bilanciamento
    balance_train = dataset_config.get("balance_train", True)
    balance_test = dataset_config.get("balance_test", False)
    
    # Batch size da training_config
    batch_size = 32  # default
    if training_config:
        batch_size = training_config.get("batch_size", 32)
    
    print(f"Client {client_id}: train={train_path}, test={test_path}, balance={balance_train}, batch_size={batch_size}")
    
    # Carica training (con bilanciamento)
    train_texts, train_labels = load_csv_pairs(train_path, balance=balance_train)
    if len(train_texts) == 0:
        raise ValueError(f"Nessun dato di training per client {client_id}")
    
    trainloader = create_dataloader(train_texts, train_labels, batch_size=batch_size, shuffle=True)
    
    # Libera memoria dei testi originali
    del train_texts, train_labels
    
    # Carica test (senza bilanciamento di default)
    test_texts, test_labels = load_csv_pairs(test_path, balance=balance_test)
    testloader = create_dataloader(test_texts, test_labels, batch_size=batch_size, shuffle=False)
    
    # Libera memoria dei testi originali
    del test_texts, test_labels
    
    # Forza garbage collection
    import gc
    gc.collect()
    
    return trainloader, testloader


def set_parameters(model, parameters):
    """Imposta i parametri del modello da una lista di array NumPy."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model):
    """Ottiene i parametri del modello come lista di array NumPy."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def train(model, trainloader, epochs=1, device=DEVICE, lr=5e-6, weight_decay=0.01, max_grad_norm=1.0):
    """Addestra il modello BERT."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (input_ids, labels) in enumerate(trainloader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            epoch_loss += loss.item()
            total_batches += 1
            
            # Libera memoria GPU
            del outputs, loss
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_epoch_loss = epoch_loss / len(trainloader)
        total_loss += epoch_loss
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_epoch_loss:.4f}")
    
    return total_loss / total_batches


def test(model, testloader, device=DEVICE):
    """Valuta il modello sul test set."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(testloader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Libera memoria
            del outputs, loss
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0
    
    # Metriche dettagliate
    cr = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "loss": avg_loss,
        # Macro metrics
        "macro_f1": cr.get("macro avg", {}).get("f1-score", 0.0),
        "macro_precision": cr.get("macro avg", {}).get("precision", 0.0),
        "macro_recall": cr.get("macro avg", {}).get("recall", 0.0),
        # Micro/Weighted metrics
        "micro_f1": cr.get("weighted avg", {}).get("f1-score", 0.0),
        "micro_precision": cr.get("weighted avg", {}).get("precision", 0.0),
        "micro_recall": cr.get("weighted avg", {}).get("recall", 0.0),
    }
    
    return avg_loss, accuracy, metrics


def weighted_average(metrics):
    """Aggregazione ponderata delle metriche."""
    if not metrics:
        return {}
    
    # Metriche da escludere dall'aggregazione
    exclude_keys = {"client_id"}
    
    totals = {}
    weights = {}
    
    for num_examples, metrics_dict in metrics:
        for key, value in metrics_dict.items():
            if key in exclude_keys:
                continue
            try:
                val = float(value)
                if key not in totals:
                    totals[key] = 0.0
                    weights[key] = 0
                totals[key] += val * num_examples
                weights[key] += num_examples
            except (TypeError, ValueError):
                continue
    
    return {k: totals[k] / weights[k] for k in totals if weights[k] > 0}
