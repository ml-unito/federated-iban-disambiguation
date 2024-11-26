
import torch
import pandas as pd
from fluke.data.datasets import DataContainer
from lib.CharacterBertForClassification import *
from sklearn.model_selection import train_test_split


# DATASET_PATH = "./Dataset_federated_learning/dataset_26k.csv"
DATASET_PATH = "./Dataset_federated_learning/dataset_1000k.csv"
# DATASET_PATH = "./Dataset_federated_learning/dataset_1k.csv"



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



def MyDataset() -> DataContainer:
    
    
    dataset = load_dataset()
    tokenized_texts = tokenize_dataset(dataset)
    input_tensors, labels = lookup_table(tokenized_texts, dataset)
    labels = labels.unsqueeze(1)
    labels = labels.float()

    X_train, X_test, y_train, y_test = train_test_split(input_tensors, labels, train_size=0.8, random_state=42, stratify=labels)

    return DataContainer(X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        num_classes=2)