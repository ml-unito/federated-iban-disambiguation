import pandas as pd
from sklearn.model_selection import train_test_split
from lib.datasetManipulation import *
import sys

DIR_PATH = "./dataset/"
DATASET_PATH = "benchmark_intesa.csv"
SAVE_DATASETS = True



def test_datasets(datasets: list, prob_shared: float):
  for dataset in datasets:
    current_prob_shared = dataset.groupby('IsShared').size()[1] * 100 / len(dataset)
    if not (current_prob_shared >= (prob_shared - 5) and current_prob_shared <= (prob_shared + 5)):
      return 0
  
  return 1



def create_couple_dataset(dataset: str, output_path: str, to_save=False):
    dataset = prepocess_dataset(dataset)
    dataset = balance_dataset(dataset, "IsShared")
    dataset = create_pairs(dataset)
    dataset = balance_dataset(dataset, "label")
    
    dataset.to_csv(output_path, index=False)

    return dataset



def split_dataset(dataset: pd.DataFrame, train_size: float):
  iban_list = dataset.AccountNumber.unique()
  isShared = dataset.groupby('AccountNumber', sort=False)['IsShared'].first().loc[iban_list].values
  x_iban_list, y_iban_list = train_test_split(iban_list, train_size=train_size, stratify=isShared)
  
  df_x = dataset.loc[dataset.AccountNumber.isin(x_iban_list)]
  df_y = dataset.loc[dataset.AccountNumber.isin(y_iban_list)]

  return df_x, df_y



def generate_datasets():
  dataset = pd.read_csv(DIR_PATH + DATASET_PATH, index_col=0)
  dataset["Name"] = dataset["Name"].str.lower()
  dataset["Holder"] = dataset["Holder"].str.lower()

  generating = True
  num_generation = 0
  print("Generating datasets ...")
  while generating or num_generation <= 10:
    num_generation += 1

    df_train_fl, df_test_clustering = split_dataset(dataset, train_size=0.8)
    df_clients, df_server = split_dataset(df_train_fl, train_size=0.8)

    # Split dataset between four client, with ratio 40-40-10-10
    x,y = split_dataset(df_clients, 0.8)
    df_client1, df_client2 = split_dataset(x, 0.5)
    df_client3, df_client4 = split_dataset(y, 0.5)
    
    # Test datasets generated: if they not respects shared proportion, the new datasets will generate.
    is_correct = test_datasets(datasets=[df_client1, df_client2, df_client3, df_client4, df_server, df_test_clustering],
                  prob_shared=dataset.groupby('IsShared').size()[1]*100/len(dataset))

    if is_correct:
      generating = False
      print(f"\nclient1:\tlen {len(df_client1)}\t prob shared {round(df_client1.groupby('IsShared').size()[1] * 100 / len(df_client1), 2)}")
      print(f"client2:\tlen {len(df_client2)}\t prob shared {round(df_client2.groupby('IsShared').size()[1] * 100 / len(df_client2),2)}")
      print(f"client3:\tlen {len(df_client3)}\t\t prob shared {round(df_client3.groupby('IsShared').size()[1] * 100 / len(df_client3), 2)}")
      print(f"client4:\tlen {len(df_client4)}\t\t prob shared {round(df_client4.groupby('IsShared').size()[1] * 100 / len(df_client4), 2)}")
      print(f"server:\tlen {len(df_server)}\t\t\t prob shared {round(df_server.groupby('IsShared').size()[1] * 100 / len(df_server), 2)}")
      print(f"clustering:\tlen {len(df_server)}\t\t prob shared {round(df_server.groupby('IsShared').size()[1] * 100 / len(df_server), 2)}\n")
      
      # Save all dataset generated
      if SAVE_DATASETS:
        df_client1.to_csv(DIR_PATH + "client1_train.csv")
        df_client2.to_csv(DIR_PATH + "client2_train.csv")
        df_client3.to_csv(DIR_PATH + "client3_train.csv")
        df_client4.to_csv(DIR_PATH + "client4_train.csv")
        df_server.to_csv(DIR_PATH + "server_test.csv")
        df_test_clustering.to_csv(DIR_PATH + "test_clustering.csv")
        print("Datasets saved into " + DIR_PATH + " directory.")



def generate_couple_datasets():
  # Create couple from dataset clients and server
  df_client1 = pd.read_csv(DIR_PATH + "client1_train.csv", index_col=0)
  df_client2 = pd.read_csv(DIR_PATH + "client2_train.csv", index_col=0)
  df_client3 = pd.read_csv(DIR_PATH + "client3_train.csv", index_col=0)
  df_client4 = pd.read_csv(DIR_PATH + "client4_train.csv", index_col=0)
  df_server = pd.read_csv(DIR_PATH + "server_test.csv", index_col=0)

  print("Creating couple datasets ...")

  create_couple_dataset(df_client1, DIR_PATH + "client1_train_couple.csv")
  create_couple_dataset(df_client2, DIR_PATH + "client2_train_couple.csv")
  create_couple_dataset(df_client3, DIR_PATH + "client3_train_couple.csv")
  create_couple_dataset(df_client4, DIR_PATH + "client4_train_couple.csv")
  create_couple_dataset(df_server, DIR_PATH + "server_test_couple.csv")
  
  print("Couple datasets is created.")



def main(action: str):
  if action == "GENERATE_DATASETS":
    generate_datasets()
  elif action == "GENERATE_COUPLE_DATASETS":
    generate_couple_datasets()

  

if __name__ == "__main__":
  if len(sys.argv) >= 1:
    action = sys.argv[1]
    main(action)
  else:
    print("error parameters")
