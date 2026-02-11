import copy
import sys
import pandas as pd
import json
import yaml
import os
import time
from typing import Optional, Tuple
from fluke import DDict
import fluke.utils.log as log
from fluke import FlukeENV
from fluke.data import DataSplitter
from fluke.algorithms.fedavg import FedAVG
from fluke.data.datasets import DataContainer, DummyDataContainer, FastDataLoader
from fluke.evaluation import ClassificationEval
from lib.download import download_pre_trained_model
from sklearn.model_selection import train_test_split
from datetime import datetime
from transformers import BertTokenizer
from lib.datasetManipulation import *
from rich.progress import track

from lib.kernel_sim_data_utils import load_sim_data
from sklearn.preprocessing import MinMaxScaler

#from lib.CBertClassif import *
from lib.CBertClassifFrz import *
# from lib.CBertClassifFrzSep import *


with open('./config/fl_parameters.json', "r") as data_file:
	fl_parameters = json.load(data_file)


EXP_PATH = fl_parameters["config"]["exp_path"]
ALG_PATH = fl_parameters["config"]["alg_path"]

SAVE_MODELS = fl_parameters["save_models"]
PATH_SAVE_MODELS = fl_parameters["path_save_models"]
DATE_AND_TIME = str(datetime.now()).split(".")[0].replace(" ", "_")

DIR_DATASET_PATH = fl_parameters["dir_dataset_path"]
TRAIN_PATH = fl_parameters["train_path"]
TEST_PATH = fl_parameters["test_path"]


def extract_x_and_y(dataset: pd.DataFrame, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
  tokenized_texts = tokenize_dataset(dataset, tokenizer)
  x, y = lookup_table(tokenized_texts, dataset)
  return x, y


def create_couple_df(df_path: str, balance: bool=False) -> pd.DataFrame:
  df = pd.read_csv(df_path)

  # if balance:
  #   df = balance_dataset(df, "IsShared")
  
  cp_df = create_pairs(df)
  
  if balance:
    cp_df = balance_dataset(cp_df, "label", oversample=True)

  return cp_df


def create_dc_cbert(train_path: str, test_path: str) -> DataContainer:  
  # Loads tokenizer
  tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')

  cp_train_df = create_couple_df(df_path=train_path, balance=True)
  cp_test_df = create_couple_df(df_path=test_path, balance=False)

  X_train, y_train = extract_x_and_y(cp_train_df, tokenizer)
  X_test, y_test = extract_x_and_y(cp_test_df, tokenizer)

  return DataContainer(X_train, y_train, X_test, y_test, 2)


def create_ddc_cbert(clients: int, train_path: str, test_path: str, client_test=False) -> DummyDataContainer:
  # Loads tokenizer
  tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')

  # Loads client datasets and server dataset
  cp_df_clients = [create_couple_df(df_path=train_path % (i), balance=True) for i in range(1, clients+1)]
  cp_df_server = create_couple_df(df_path=test_path, balance=False)

  # Creates FastDataLoader for each client data
  fdl_clts = []
  for df_client in track(cp_df_clients, description="Loading clients datasets"):
    x, y = extract_x_and_y(df_client, tokenizer)
    fdl = FastDataLoader(x, y, num_labels=2, batch_size=512)
    fdl_clts.append(fdl)
  
  # Creates FastDataLoader for server data
  x, y = extract_x_and_y(cp_df_server, tokenizer)
  fdl_srv = FastDataLoader(x, y, num_labels=2, batch_size=512)

  return DummyDataContainer(clients_tr=fdl_clts, 
                              clients_te= [fdl_srv]*clients if client_test else [None]*clients, 
                              server_data=fdl_srv, 
                              num_classes=2)


def create_dc_kernel(sim_train_path: str, sim_test_path: str, seed: int, bert: bool, client: int=None) -> DataContainer:
  train, test = load_sim_data(
      train_path=sim_train_path % (seed, "_w-bert" if bert else "") if client is None else sim_train_path % (client, seed, "_w-bert" if bert else ""),
      test_path=sim_test_path % (seed, "_w-bert" if bert else "")
  )
  
  scaler = MinMaxScaler()
  train.iloc[:, :-1] = scaler.fit_transform(train.iloc[:, :-1])
  test.iloc[:, :-1] = scaler.transform(test.iloc[:, :-1])

  # Convert data to PyTorch tensors
  train_x = torch.tensor(train.iloc[:, :-1].values, dtype=torch.float32)
  train_y = torch.tensor(train.iloc[:, -1].values, dtype=torch.long)
  test_x = torch.tensor(test.iloc[:, :-1].values, dtype=torch.float32)
  test_y = torch.tensor(test.iloc[:, -1].values, dtype=torch.long)

  return DataContainer(train_x, train_y, test_x, test_y, 2)


def create_ddc_kernel(clients: int, sim_train_path: str, sim_test_path: str, seed: int, bert: bool, client_test: bool=False) -> DummyDataContainer:
  df_clients = [pd.read_csv(sim_train_path % (n, seed, "_w-bert" if bert else "")) for n in range(1,clients+1)]
  df_server = pd.read_csv(sim_test_path % (seed, "_w-bert" if bert else ""))

  # Creates FastDataLoader for each client data
  fdl_clts = []
  scaler = MinMaxScaler()
  for df_client in df_clients:
    df_client.iloc[:, :-1] = scaler.fit_transform(df_client.iloc[:, :-1])

    x = torch.tensor(df_client.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df_client.iloc[:, -1].values, dtype=torch.long)
    
    fdl = FastDataLoader(x, y, num_labels=2, batch_size=1024)
    fdl_clts.append(fdl)
  
  # Creates FastDataLoader for server data
  df_server.iloc[:, :-1] = scaler.fit_transform(df_server.iloc[:, :-1])

  x = torch.tensor(df_server.iloc[:, :-1].values, dtype=torch.float32)
  y = torch.tensor(df_server.iloc[:, -1].values, dtype=torch.long)
  fdl_srv = FastDataLoader(x, y, num_labels=2, batch_size=1024)

  return DummyDataContainer(clients_tr=fdl_clts,
                            clients_te= [fdl_srv]*clients if client_test else [None]*clients, 
                            server_data=fdl_srv, 
                            num_classes=2)


def load_parameters(exp_path: str, alg_path: str) -> Tuple[DDict, DDict]:
  config_file_exp = open(exp_path)
  config_exp = yaml.safe_load(config_file_exp)

  config_file_alg = open(alg_path)
  config_alg = yaml.safe_load(config_file_alg)

  return DDict(config_exp), DDict(config_alg)


def main(log_name: str):
  config_exp, config_alg = load_parameters(EXP_PATH, ALG_PATH)

  settings = FlukeENV()
  settings.set_seed(config_exp["exp"]["seed"])
  settings.set_device(config_exp["exp"]["device"]) 

  datasets = create_ddc_cbert(num_clients=config_exp["protocol"]["n_clients"], 
                                         train_path=TRAIN_PATH, test_path=TEST_PATH, dir_dataset_path=DIR_DATASET_PATH, 
                                         client_test=True)

  settings.set_evaluator(ClassificationEval(eval_every=1, n_classes=datasets.num_classes))
  settings.set_eval_cfg(config_exp["eval"])

  algorithm = FedAVG(n_clients=config_exp["protocol"]["n_clients"],
                    data_splitter=DataSplitter(dataset=datasets, **config_exp["data"]),
                    hyper_params=DDict(**config_alg["hyperparameters"]) )

  logger = log.get_logger(config_exp["logger"]["name"], name=log_name, **config_exp["logger"].exclude("name"))
  cfg = copy.copy(config_exp)
  cfg.update(config_alg)
  logger.init(**cfg)
  
  algorithm.set_callbacks(logger)
  
  start_time = time.time()
  algorithm.run(n_rounds=config_exp["protocol"]["n_rounds"], eligible_perc=config_exp["protocol"]["eligible_perc"])
  end_time = time.time()

  if SAVE_MODELS:
    algorithm.save(PATH_SAVE_MODELS + DATE_AND_TIME)
    print("\nClients model and server model are saved in \"" + PATH_SAVE_MODELS + DATE_AND_TIME + " directory.\n")

  # Adds information into log
  general_info_log = {}
  for i, df_client in enumerate(datasets.clients_tr, start=1):
    general_info_log["Couples number in Client" + str(i) + " dataset"] = df_client.size
  general_info_log["Couples number in Server dataset"] = datasets.server_data.size
  general_info_log["Execution time (sec)"] = round(end_time - start_time, 5)
  logger.pretty_log(data=general_info_log, title="General information")
  
  # Save log in json file
  logger.save("./out/federated_learning_logs/log_fl_" + str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + ".json")
  

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("USAGE: python3 federated_learning.py LOG_NAME")
    exit()

  log_name = sys.argv[1]

  download_pre_trained_model()
  main(log_name)
