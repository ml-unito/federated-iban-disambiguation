
import torch
import pandas as pd
import json
from fluke import DDict
from fluke.utils.log import Log
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
DEVICE = "cuda" if fl_parameters["device"] == "cuda" and torch.cuda.is_available() else 'cpu'
NUM_CLIENT = 4



def extract_x_and_y(dataset: pd.DataFrame):
  tokenized_texts = tokenize_dataset(dataset)
  x, y = lookup_table(tokenized_texts, dataset)
  y = y.unsqueeze(1)
  y = y.float()

  return x, y


def create_dummy_data_container() -> DummyDataContainer:
  # Loads client datasetS and server dataset
  df_clients = [pd.read_csv(DIR_DATASET_PATH + "client" + str(i) + "_train_couple.csv") for i in range(1, NUM_CLIENT+1)]
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

  return DummyDataContainer(clients_tr=fdl_clients, clients_te=[None]*NUM_CLIENT, 
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


def load_dataset():
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



def main():
  settings = GlobalSettings()
  settings.set_seed(42)         # we set a seed for reproducibility
  settings.set_device(DEVICE) 
  
  # Load datasets
  # dataset = load_dataset()
  # tokenized_texts = tokenize_dataset(dataset)
  # input_tensors, labels = lookup_table(tokenized_texts, dataset)
  # labels = labels.unsqueeze(1)
  # labels = labels.float()
  # dataset = MyDataset(input_tensors, labels)

  datasets = create_dummy_data_container()

  # we set the evaluator to be used by both the server and the clients
  settings.set_evaluator(ClassificationEval(eval_every=1, n_classes=datasets.num_classes))
  splitter = DataSplitter(dataset=datasets, distribution="iid")
  
  
  client_hp = DDict(
      batch_size=512,
      local_epochs=5,
      loss="BCELoss",
      optimizer=DDict(
        name="AdamW",
        lr=0.0001,
        weight_decay=0.001),
      scheduler=DDict(
        name= "StepLR",
        gamma= 0.995,
        step_size= 10),
        testset_path="./Dataset_federated_learning/test_sets"
  )

  hyperparams = DDict(client=client_hp,
                      server=DDict(weighted=True),
                      model=CharacterBertForClassification()) 

  
  # algorithm = MyFLClustering(n_clients=4,
  #                   data_splitter=splitter,
  #                   hyper_params=hyperparams)

  algorithm = FedAVG(n_clients=NUM_CLIENT,
                    data_splitter=splitter,
                    hyper_params=hyperparams)

  logger = Log()
  algorithm.set_callbacks(logger)
  algorithm.run(n_rounds=3, eligible_perc=1)

  logger.save("./out/log_fl_" + str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + ".json")
  


if __name__ == "__main__":
  main()

