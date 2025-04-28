import pandas as pd
from sklearn.model_selection import train_test_split
from lib.datasetManipulation import *
import typer

app = typer.Typer()

DIR_PATH = "./dataset/"
DATASET_PATH = "benchmark_intesa.csv"
SAVE_DATASETS = True



def test_datasets(datasets: list, prob_shared: float):
  for dataset in datasets:
    current_prob_shared = dataset[dataset['IsShared'] == 1]['AccountNumber'].nunique() / len(set(dataset["AccountNumber"])) * 100 #dataset.groupby('IsShared').size()[1] * 100 / len(dataset)
    if not (current_prob_shared >= (prob_shared - 3) and current_prob_shared <= (prob_shared + 3)):
      return 0
  
  return 1


def split_dataset(dataset: pd.DataFrame, train_size: float, random_state=42):
  iban_list = dataset.AccountNumber.unique()
  isShared = dataset.groupby('AccountNumber', sort=False)['IsShared'].first().loc[iban_list].values
  x_iban_list, y_iban_list = train_test_split(iban_list, train_size=train_size, stratify=isShared, random_state=random_state)
  
  df_x = dataset.loc[dataset.AccountNumber.isin(x_iban_list)]
  df_y = dataset.loc[dataset.AccountNumber.isin(y_iban_list)]

  return df_x, df_y


def create_couple_dataset(dataset: str, output_path: str, to_save=False):
    dataset = prepocess_dataset(dataset)
    dataset = balance_dataset(dataset, "IsShared")
    dataset = create_pairs(dataset)
    dataset = balance_dataset(dataset, "label")
    
    dataset.to_csv(output_path, index=False)

    return dataset


def get_shared_prob(df):
  return round(df[df['IsShared'] == 1]['AccountNumber'].nunique() / len(set(df["AccountNumber"])) * 100, 2)


def create_train_test_split(dataset, seed=42):
  # Split dataset for train and test (80-20)
  df_train, df_test = split_dataset(dataset, train_size=0.8, random_state=seed)

  # Split dataset between four client, with ratio 40-40-10-10
  x,y = split_dataset(df_train, 0.8, random_state=seed)
  df_client1, df_client2 = split_dataset(x, 0.5, random_state=seed)
  df_client3, df_client4 = split_dataset(y, 0.5, random_state=seed)

  # Test datasets generated on shared proportion
  is_correct = test_datasets(datasets=[df_test, df_train, df_client1, df_client2, df_client3, df_client4],
                prob_shared=get_shared_prob(dataset))
  if not is_correct:
    print("Error of split generation: shared proportion is not respected.")
    exit()
  
  print(f"Original dataset:\tlen {len(dataset)}\t\t\t prob shared {get_shared_prob(dataset)}")
  print(f"\nclient1:\tlen {len(df_client1)}\t prob shared {get_shared_prob(df_client1)}")
  print(f"client2:\tlen {len(df_client2)}\t prob shared {get_shared_prob(df_client2)}")
  print(f"client3:\tlen {len(df_client3)}\t\t prob shared {get_shared_prob(df_client3)}")
  print(f"client4:\tlen {len(df_client4)}\t\t prob shared {get_shared_prob(df_client4)}")
  print(f"test:\t\tlen {len(df_test)}\t prob shared {get_shared_prob(df_test)}")
  print(f"train:\t\tlen {len(df_train)}\t prob shared {get_shared_prob(df_train)}")
  
  # Save all dataset generated
  if SAVE_DATASETS:
    df_client1.to_csv(DIR_PATH + "split_dataset/client1_train.csv")
    df_client2.to_csv(DIR_PATH + "split_dataset/client2_train.csv")
    df_client3.to_csv(DIR_PATH + "split_dataset/client3_train.csv")
    df_client4.to_csv(DIR_PATH + "split_dataset/client4_train.csv")
    df_test.to_csv(DIR_PATH + "split_dataset/df_test.csv")
    df_train.to_csv(DIR_PATH + "split_dataset/df_train.csv")
    print("Datasets saved into " + DIR_PATH + "split_dataset/ directory.")


def generate_couple_datasets():
  # Create couple from dataset clients and server
  df_client1 = pd.read_csv(DIR_PATH + "client1_train.csv", index_col=0)
  df_client2 = pd.read_csv(DIR_PATH + "client2_train.csv", index_col=0)
  df_client3 = pd.read_csv(DIR_PATH + "client3_train.csv", index_col=0)
  df_client4 = pd.read_csv(DIR_PATH + "client4_train.csv", index_col=0)
  df_test = pd.read_csv(DIR_PATH + "df_test.csv", index_col=0)
  df_train = pd.read_csv(DIR_PATH + "df_train.csv", index_col=0)

  print("Creating couple datasets ...")

  create_couple_dataset(df_client1, DIR_PATH + "client1_train_couple.csv")
  create_couple_dataset(df_client2, DIR_PATH + "client2_train_couple.csv")
  create_couple_dataset(df_client3, DIR_PATH + "client3_train_couple.csv")
  create_couple_dataset(df_client4, DIR_PATH + "client4_train_couple.csv")
  create_couple_dataset(df_test, DIR_PATH + "test_couple.csv")
  create_couple_dataset(df_train, DIR_PATH + "train_couple.csv")
  
  print("Couple datasets is created.")


@app.command()
def split(seed: int = 42):
  dataset = pd.read_csv(DIR_PATH + DATASET_PATH, index_col=0)
  create_train_test_split(dataset, seed=seed)

  

if __name__ == "__main__":
  app()
