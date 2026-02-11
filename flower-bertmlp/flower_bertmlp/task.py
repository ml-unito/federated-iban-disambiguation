"""flower-bertmlp: A Flower / PyTorch app."""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import OrderedDict
from transformers import BertTokenizer

# Importa le funzioni da CBertClassif
from lib.CBertClassif import (
    CBertClassif as CharacterBertForClassification,
    train,
    test,
    indexer,
    lookup_table
)

from lib.download import *

# Importa funzioni per manipolazione dataset
from lib.datasetManipulation import create_pairs, balance_dataset, tokenize_dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SYSTEM_SEED = 12345  # Seed per replicabilità del sistema (modello, CUDA, ecc.)

download_pre_trained_model()

tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')


def set_system_seed(seed: int = SYSTEM_SEED):
    
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
    
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def load_data(client_id, data_config, training_config):
    
    dataset_config = data_config["dataset"]
    
    # Determina il path di training per questo client
    client_train_paths = dataset_config["client_train_paths"]
    
    client_train_paths = [path.replace("split_dataset", f"split_dataset_{dataset_config['seed']}") for path in client_train_paths]
    
    idx = client_id - 1  # client_id è 1-based
    
    if idx >= len(client_train_paths):
        raise ValueError(f"Client ID {client_id} fuori range (max: {len(client_train_paths)})")
    
    train_path = client_train_paths[idx]
    test_path = dataset_config["test_path"]
    
    test_path = test_path.replace("split_dataset", f"split_dataset_{dataset_config['seed']}")
    
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
    X_train = tokenize_dataset(train_df, tokenizer=tokenizer)
    X_val = tokenize_dataset(test_df, tokenizer=tokenizer)
    
    # Labels come liste
    y_train = train_df["label"].tolist()
    y_val = test_df["label"].tolist()
    
    del train_df, test_df
    
    batch_size = training_config.get("batch_size", 16)
    
    print(f"Client {client_id}: train={len(X_train)} coppie, val={len(X_val)} coppie, batch_size={batch_size}")
    
    return X_train, y_train, X_val, y_val, batch_size


def weighted_average(metrics):
    
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



########################################################
# LOCKING PER FIT CONCORRENTI

# import os
# import time
# from pathlib import Path

# LOCK_DIR = "./tmp/flower_fit_locks"  # Directory per i lock file

# def ensure_lock_dir():
#     """Crea la directory per i lock se non esiste."""
#     Path(LOCK_DIR).mkdir(exist_ok=True)

# def acquire_fit_lock(client_id, timeout=7200):
#     """
#     Acquisisci il lock: massimo 2 client possono fare fit contemporaneamente.
#     Se non disponibile, aspetta.
#     """
#     ensure_lock_dir()
#     lock_file = os.path.join(LOCK_DIR, f"client_{client_id}.lock")
#     start_time = time.time()
    
#     while True:
#         # Conta i lock file esistenti
#         existing_locks = len([f for f in os.listdir(LOCK_DIR) if f.endswith('.lock')])
        
#         if existing_locks < 2:
            
#             # fai in modo che il client 1 e due non eseguano mai assieme il fit
#             if client_id == 2 and os.path.exists(os.path.join(LOCK_DIR, "client_1.lock")):
#                 print(f"Client {client_id}: in attesa del lock del Client 1...")
#                 time.sleep(10)
#                 continue
#             if client_id == 1 and os.path.exists(os.path.join(LOCK_DIR, "client_2.lock")):
#                 print(f"Client {client_id}: in attesa del lock del Client 2...")
#                 time.sleep(10)
#                 continue 
            
#             # C'è spazio, scrivi il lock
#             with open(lock_file, 'w') as f:
#                 f.write(str(client_id))
#             print(f"Client {client_id}: LOCK ACQUISITO ({existing_locks + 1}/2 in uso)")
#             return lock_file
        
#         # Timeout
#         if time.time() - start_time > timeout:
#             raise TimeoutError(f"Client {client_id}: timeout in attesa del lock")
        
#         print(f"Client {client_id}: in attesa... ({existing_locks}/2 lock in uso)")
#         time.sleep(10)  # Aspetta 1 secondo prima di ritentare

# def release_fit_lock(lock_file):
#     """Rilascia il lock eliminando il file."""
#     if os.path.exists(lock_file):
#         os.remove(lock_file)
#         client_id = os.path.basename(lock_file).split('_')[1].split('.')[0]
#         print(f"Client {client_id}: LOCK RILASCIATO")
