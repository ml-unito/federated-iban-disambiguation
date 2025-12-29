"""flower-bertmlp: A Flower / PyTorch app."""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import OrderedDict

# Importa le funzioni da CBertClassif
from lib.CBertClassif import (
    CBertClassif as CharacterBertForClassification,
    train,
    test,
    indexer,
    lookup_table
)

# Importa funzioni per manipolazione dataset
from lib.datasetManipulation import create_pairs, balance_dataset, tokenize_dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SYSTEM_SEED = 12345  # Seed per replicabilità del sistema (modello, CUDA, ecc.)


def set_system_seed(seed: int = SYSTEM_SEED):
    """Fissa tutti i seed di sistema per garantire replicabilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Per multi-GPU
    
    # Rende le operazioni CUDA deterministiche
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Variabile d'ambiente per hash deterministico
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_parameters(model):
    """Estrae i parametri del modello come lista di numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Imposta i parametri del modello da lista di numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def load_data(client_id, data_config, training_config):
    """Carica i dati per un client specifico. Restituisce liste raw per CBertClassif."""
    dataset_config = data_config["dataset"]
    
    # Determina il path di training per questo client
    client_train_paths = dataset_config["client_train_paths"]
    idx = client_id - 1  # client_id è 1-based
    
    if idx >= len(client_train_paths):
        raise ValueError(f"Client ID {client_id} fuori range (max: {len(client_train_paths)})")
    
    train_path = client_train_paths[idx]
    test_path = dataset_config["test_path"]
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Verifica se il CSV è già preprocessato (ha colonna 'text') o è raw
    if 'text' not in train_df.columns:
        # CSV raw: crea coppie
        print(f"Client {client_id}: creazione coppie da dati raw (train)...")
        train_df = create_pairs(train_df)
        if dataset_config.get("balance_train", True):
            print(f"Client {client_id}: bilanciamento dataset train...")
            train_df = balance_dataset(train_df, "label", oversample=True)
    
    if 'text' not in test_df.columns:
        print(f"Client {client_id}: creazione coppie da dati raw (test)...")
        test_df = create_pairs(test_df)
        if dataset_config.get("balance_test", False):
            test_df = balance_dataset(test_df, "label", oversample=True)
    
    # Tokenizza i testi (lista di liste di token)
    X_train = tokenize_dataset(train_df, tokenizer=None)
    X_val = tokenize_dataset(test_df, tokenizer=None)
    
    # Labels come liste
    y_train = train_df["label"].tolist()
    y_val = test_df["label"].tolist()
    
    batch_size = training_config.get("batch_size", 16)
    
    print(f"Client {client_id}: train={len(X_train)} coppie, val={len(X_val)} coppie, batch_size={batch_size}")
    
    return X_train, y_train, X_val, y_val, batch_size


def weighted_average(metrics):
    """Aggregazione ponderata delle metriche."""
    if not metrics:
        return {}
    
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
