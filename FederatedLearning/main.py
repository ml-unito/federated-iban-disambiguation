from fluke.data import DataSplitter
from fluke import DDict
from fluke.utils.log import Log
from fluke.evaluation import ClassificationEval
from fluke import GlobalSettings
from fluke.algorithms.fedavg import FedAVG
from model import MyMLP
import pandas as pd
from fluke.data.datasets import DataContainer
import torch

DATASET_PATH = "./dataset/dataset_09-10-2024_12-17-09.csv"



def MyDataset() -> DataContainer:
    # Random dataset with 100 2D points from 2 classes
    X = torch.randn(100, 2)
    y = torch.randint(0, 2, (100,))

    return DataContainer(X_train=X[:80],
                         y_train=y[:80],
                         X_test=X[80:],
                         y_test=y[80:],
                         num_classes=2)


def load_dataset():
  dataset = pd.read_csv(DATASET_PATH)

  return dataset



def main():
  settings = GlobalSettings()
  settings.set_seed(42) # we set a seed for reproducibility
  settings.set_device("cpu") # we use the CPU for this example


  dataset = MyDataset()

  # we set the evaluator to be used by both the server and the clients
  settings.set_evaluator(ClassificationEval(eval_every=1, n_classes=dataset.num_classes))

  splitter = DataSplitter(dataset=dataset,
                          distribution="iid")
  
  
  client_hp = DDict(
      batch_size=10,
      local_epochs=5,
      loss="CrossEntropyLoss",
      optimizer=DDict(
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001),
      scheduler=DDict(
        gamma=1,
        step_size=1)
  )

  hyperparams = DDict(client=client_hp,
                      server=DDict(weighted=True),
                      model=MyMLP()) # we use our network :)


  algorithm = FedAVG(n_clients=2,
                    data_splitter=splitter,
                    hyper_params=hyperparams)

  logger = Log()
  algorithm.set_callbacks(logger)


  algorithm.run(n_rounds=10, eligible_perc=0.5)



if __name__ == "__main__":
  main()

