
import torch
import pandas as pd
import json
import yaml
import time
from fluke import DDict
import fluke.utils.log as log
from fluke import GlobalSettings
from fluke.data import DataSplitter
from fluke.algorithms.fedavg import FedAVG
from lib.MyFLClustering import MyFLClustering
from fluke.data.datasets import DataContainer, DummyDataContainer, FastDataLoader
from fluke.evaluation import ClassificationEval
from lib.download import download_pre_trained_model
from sklearn.model_selection import train_test_split
from datetime import datetime

download_pre_trained_model()
from lib.CharacterBertForClassification import *


with open('./config/fl_parameters.json', "r") as data_file:
	fl_parameters = json.load(data_file)


# DATASET_PATH = "./test_federated_learning_dataset.csv"
# DATASET_PATH = "./dataset_prova.csv"
# DATASET_PATH = "./Dataset_federated_learning/dataset_1k.csv"
DIR_DATASET_PATH = fl_parameters["dir_dataset_path"]
DATASET_PATH = fl_parameters["dataset_path"]
COUPLE_DATASET_PATH = fl_parameters["couple_dataset_path"]
EXP_PATH = fl_parameters["config"]["exp_path"]
ALG_PATH = fl_parameters["config"]["alg_path"]
SAVE_MODELS = fl_parameters["save_models"]
PATH_SAVE_MODELS = fl_parameters["path_save_models"]



def extract_x_and_y(dataset: pd.DataFrame) -> list:
  tokenized_texts = tokenize_dataset(dataset)
  x, y = lookup_table(tokenized_texts, dataset)
  y = y.unsqueeze(1)
  y = y.float()

  return x, y


def create_dummy_data_container(num_clients: int) -> DummyDataContainer:
  # Loads client datasetS and server dataset
  df_clients = [pd.read_csv(DIR_DATASET_PATH + "client" + str(i) + "_train_couple.csv") for i in range(1, num_clients+1)]
  df_server = pd.read_csv(DIR_DATASET_PATH + "server_test_couple.csv")

  # Creates FastDataLoader for each client data
  fdl_clients = []
  for df_client in df_clients:
    x, y = extract_x_and_y(df_client)
    fdl = FastDataLoader(x, y, num_labels=2, batch_size=512)
    fdl_clients.append(fdl)
  
  # Creates FastDataLoader for server data
  x, y = extract_x_and_y(df_server)
  fdl_server = FastDataLoader(x, y, num_labels=2, batch_size=512)

  return DummyDataContainer(clients_tr=fdl_clients, clients_te=[None]*num_clients, 
                            server_data=fdl_server, 
                            num_classes=2)



def MyDataset(X, y) -> DataContainer:
  """ My dataset container """
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
  return DataContainer(X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        num_classes=2)



def load_dataset() -> pd.DataFrame:
  dataset = pd.read_csv(DATASET_PATH)
  return dataset



def tokenize_dataset(dataframe):
  """ 
      Tokenize the dataset for the encoding layer of the CharacterBERT model.
      The tokenization is done with the symbol '@' to separate the names.
  """
  return dataframe['text'].apply(lambda x: ['[CLS]', *[y.strip() for y in x.split("@")], '[SEP]'])



def lookup_table(tokenized_texts, dataframe):
  """ define the input tensors for the CharacterBert model """
  
  indexer = CharacterIndexer()
  input_tensors = indexer.as_padded_tensor(tokenized_texts)     # Create input tensor
  labels = torch.tensor(dataframe['label'].values)  
  return input_tensors, labels


def load_parameters() -> list:
  config_file_exp = open(EXP_PATH)
  config_exp = yaml.safe_load(config_file_exp)

  config_file_alg = open(ALG_PATH)
  config_alg = yaml.safe_load(config_file_alg)

  return config_exp, config_alg


def main():
  config_exp, config_alg = load_parameters()

  settings = GlobalSettings()
  settings.set_seed(config_exp["exp"]["seed"])
  settings.set_device(config_exp["exp"]["device"]) 
  
  # Load datasets
  # dataset = load_dataset()
  # tokenized_texts = tokenize_dataset(dataset)
  # input_tensors, labels = lookup_table(tokenized_texts, dataset)
  # labels = labels.unsqueeze(1)
  # labels = labels.float()
  # dataset = MyDataset(input_tensors, labels)

  datasets = create_dummy_data_container(num_clients=config_exp["protocol"]["n_clients"])

  settings.set_evaluator(ClassificationEval(eval_every=1, n_classes=datasets.num_classes))

  algorithm = FedAVG(n_clients=config_exp["protocol"]["n_clients"],
                    data_splitter=DataSplitter(dataset=datasets, **config_exp["data"]),
                    hyper_params=DDict(**config_alg["hyperparameters"]) )

  logger = log.get_logger(**config_exp["logger"])
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
  main()

