
import torch
import pandas as pd
from fluke import DDict
from fluke.utils.log import Log
from fluke import GlobalSettings
from fluke.data import DataSplitter
# from fluke.algorithms.fedavg import FedAVG
from lib.MyFLClustering import MyFLClustering
from fluke.data.datasets import DataContainer
from fluke.evaluation import ClassificationEval
from lib.download import download_pre_trained_model
from sklearn.model_selection import train_test_split


download_pre_trained_model()
from lib.CharacterBertForClassification import *



# DATASET_PATH = "./test_federated_learning_dataset.csv"
# DATASET_PATH = "./dataset_prova.csv"
DATASET_PATH = "./Dataset_federated_learning/dataset_1k.csv"


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
  settings.set_device("cuda:0")    # we use the CPU for this example
  dataset = load_dataset()
  tokenized_texts = tokenize_dataset(dataset)
  input_tensors, labels = lookup_table(tokenized_texts, dataset)
  labels = labels.unsqueeze(1)
  labels = labels.float()
  dataset = MyDataset(input_tensors, labels)

  # we set the evaluator to be used by both the server and the clients
  settings.set_evaluator(ClassificationEval(eval_every=1, n_classes=dataset.num_classes))
  splitter = DataSplitter(dataset=dataset, distribution="iid")
  
  
  client_hp = DDict(
      batch_size=512,
      local_epochs=1,
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
                      model=CharacterBertForClassification())   # we use our network :)

  
  algorithm = MyFLClustering(n_clients=10,
                    data_splitter=splitter,
                    hyper_params=hyperparams)

  logger = Log()
  algorithm.set_callbacks(logger)
  algorithm.run(n_rounds=3, eligible_perc=1)



if __name__ == "__main__":
  main()

