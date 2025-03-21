import copy
import sys
import pandas as pd
import json
import yaml
import time
from fluke import DDict
import fluke.utils.log as log
from fluke import GlobalSettings
from fluke.data import DataSplitter
from fluke.algorithms.fedavg import FedAVG
from fluke.data.datasets import DataContainer, DummyDataContainer, FastDataLoader
from fluke.evaluation import ClassificationEval
# from lib.download import download_pre_trained_model
# from sklearn.model_selection import train_test_split
from datetime import datetime
from transformers import BertTokenizer

# download_pre_trained_model()
# from lib.CharacterBertForClassificationOptimized import *
from lib.CharacterBertForClassificationOptimizedFreezed import *
# from lib.CharacterBertForClassificationOptimizedFreezedSeparated import *
from lib.datasetManipulation import *


with open('./config/fl_parameters.json', "r") as data_file:
	fl_parameters = json.load(data_file)


DIR_DATASET_PATH = fl_parameters["dir_dataset_path"]
EXP_PATH = fl_parameters["config"]["exp_path"]
ALG_PATH = fl_parameters["config"]["alg_path"]
SAVE_MODELS = fl_parameters["save_models"]
PATH_SAVE_MODELS = fl_parameters["path_save_models"]



def extract_x_and_y(dataset: pd.DataFrame, tokenizer) -> list:
  tokenized_texts = tokenize_dataset(dataset, tokenizer)
  x, y = lookup_table(tokenized_texts, dataset)
  return x, y


def create_dummy_data_container(num_clients: int, client_test=False) -> DummyDataContainer:
  # Loads client datasetS and server dataset
  df_clients = [pd.read_csv(DIR_DATASET_PATH + "client" + str(i) + "_train_couple.csv") for i in range(1, num_clients+1)]
  df_server = pd.read_csv(DIR_DATASET_PATH + "server_test_couple.csv")

  # Loads tokenizer
  tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')

  # Creates FastDataLoader for each client data
  fdl_clients = []
  for df_client in df_clients:
    x, y = extract_x_and_y(df_client, tokenizer)
    fdl = FastDataLoader(x, y, num_labels=2, batch_size=512)
    fdl_clients.append(fdl)
  
  # Creates FastDataLoader for server data
  x, y = extract_x_and_y(df_server, tokenizer)
  fdl_server = FastDataLoader(x, y, num_labels=2, batch_size=512)

  return DummyDataContainer(clients_tr=fdl_clients, 
                            clients_te= [fdl_server]*num_clients if client_test else [None]*num_clients, 
                            server_data=fdl_server, 
                            num_classes=2)


def load_parameters() -> list:
  config_file_exp = open(EXP_PATH)
  config_exp = yaml.safe_load(config_file_exp)

  config_file_alg = open(ALG_PATH)
  config_alg = yaml.safe_load(config_file_alg)

  return DDict(config_exp), DDict(config_alg)


def main(log_name: str):
  config_exp, config_alg = load_parameters()

  settings = GlobalSettings()
  settings.set_seed(config_exp["exp"]["seed"])
  settings.set_device(config_exp["exp"]["device"]) 

  datasets = create_dummy_data_container(num_clients=config_exp["protocol"]["n_clients"], client_test=True)

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
    algorithm.save("./out/federated_learning_models/")
    print("\nClients model and server model are saved in \"" + "./out/federated_learning_models\""+ " directory.\n")

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
  if len(sys.argv) < 4:
    print("USAGE: python3 federated_learning.py LOG_NAME")
    exit()

  log_name = sys.argv[1]
  main(log_name)
