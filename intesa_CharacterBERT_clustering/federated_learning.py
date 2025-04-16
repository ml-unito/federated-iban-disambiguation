import copy
import sys
import pandas as pd
import json
import yaml
import os
import time
from typing import Tuple
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

download_pre_trained_model()
# from lib.CBertClassif import *
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


def create_couple_df(df_path: str) -> pd.DataFrame:
  df = pd.read_csv(df_path)

  df = balance_dataset(df, "IsShared")
  cp_df = create_pairs(df)
  cp_df = balance_dataset(cp_df, "label")

  return cp_df


def create_dummy_data_container(num_clients: int, train_path: str, test_path: str, dir_dataset_path: str, client_test=False) -> DummyDataContainer:
  # Loads tokenizer
  tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')

  if num_clients == 1:
    # Loads datasets
    cp_train_df = create_couple_df(df_path=train_path)
    cp_test_df = create_couple_df(df_path=test_path)

    X_train, y_train = extract_x_and_y(cp_train_df, tokenizer)
    X_test, y_test = extract_x_and_y(cp_test_df, tokenizer)

    # Creates FastDataLoader for client data and server data
    fdl_clt = FastDataLoader(X_train, y_train, num_labels=2, batch_size=512)
    fdl_srv = FastDataLoader(X_test, y_test, num_labels=2, batch_size=512)

    return DummyDataContainer(clients_tr=[fdl_clt], 
                              clients_te=[fdl_srv] if client_test else [None], 
                              server_data=fdl_srv, 
                              num_classes=2)
  else:
    # Loads client datasets and server dataset
    cp_df_clients = [create_couple_df(df_path=dir_dataset_path + "client" + str(i) + "_train_pp.csv") for i in range(1, num_clients+1)]
    cp_df_server = create_couple_df(df_path=test_path)

    # Creates FastDataLoader for each client data
    fdl_clts = []
    for df_client in cp_df_clients:
      x, y = extract_x_and_y(df_client, tokenizer)
      fdl = FastDataLoader(x, y, num_labels=2, batch_size=512)
      fdl_clts.append(fdl)
    
    # Creates FastDataLoader for server data
    x, y = extract_x_and_y(cp_df_server, tokenizer)
    fdl_srv = FastDataLoader(x, y, num_labels=2, batch_size=512)

    return DummyDataContainer(clients_tr=fdl_clts, 
                              clients_te= [fdl_srv]*num_clients if client_test else [None]*num_clients, 
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

  datasets = create_dummy_data_container(num_clients=config_exp["protocol"]["n_clients"], 
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

  main(log_name)
