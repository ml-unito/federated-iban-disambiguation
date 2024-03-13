# install all necessary libraries
import os
try: import pipreqs
except ImportError as error: os.system('pip install pipreqs')
os.system("pip install -r ./requirements.txt")

import json
import string
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from faker import Faker
from datetime import datetime
from lib.permutations import generate_permutations


__name__ = "__main__"



with open('./config/parameters.json', "r") as data_file:
  parameters = json.load(data_file)

PATH_OUTPUT_FILE = parameters["path_output_file"]

# dimensione massima datasets
MAX_DIM_DATASETS = parameters["max_dim_datasets"]

# numero di iban
NUM_IBAN = parameters["num_iban"] #MAX_DIM_DATASETS // 10

# range numero entry per iban
MIN_RANGE_ENTRY = parameters["min_range_entry"]
MAX_RANGE_ENTRY = parameters["max_range_entry"]

# range titolari conto condiviso
MIN_RANGE_HOLDERS = parameters["min_range_holders"]
MAX_RANGE_HOLDERS = parameters["max_range_holders"]

T = parameters["T"]         # Temperature factor ---> controlla il fattore di distorsione di una stringa
C = parameters["C"]         # Changeable factor ---> controlla il fattore di aggiunta di spazi bianchi e romozione di parole
V = parameters["V"]         # Variability factor ---> controlla il fattore di modifica delle parti societarie
EDIT = parameters["EDIT"]   # Edit distance ---> controlla l'avvicinamento con le parti societarie a partire da una lista nota

BIC_COUNTRY_CODES = parameters["bic_country_codes"]
FAKER_COUNTRY_CODES = parameters["faker_country_codes"]



def check_parameters():
  if MAX_DIM_DATASETS <= 0:
    raise Exception("Exception: max_dim_datasets must be positive number.")
  if MAX_RANGE_ENTRY < MIN_RANGE_ENTRY:
    raise Exception("Exception: max_range_entry must be grater that min_range_entry.")
  if MIN_RANGE_ENTRY <= 0 or MAX_RANGE_ENTRY <=0:
    raise Exception("Exception: max_range_entry and min_range_entry must be positive number.")
  if MAX_RANGE_HOLDERS < MIN_RANGE_HOLDERS:
    raise Exception("Exception: max_range_holders must be grater that min_range_holders.")
  if MIN_RANGE_HOLDERS <= 0 or MAX_RANGE_HOLDERS <=0:
    raise Exception("Exception: max_range_holders and min_range_holders must be positive number.")



def bic_manual_generator():
  country_code = random.choice(BIC_COUNTRY_CODES)
  bank_code = "".join([random.choice(string.ascii_uppercase) for _ in range(4)])
  location_code = random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase+'012')
  return bank_code + country_code + location_code, country_code


def iban_generator():
  country_code = random.choice(FAKER_COUNTRY_CODES)
  fake = Faker(country_code)
  return fake.iban()


def company_generator(country_code):
  fake = Faker(country_code)
  return fake.company()


def address_generator(country_code):
  fake = Faker(country_code)
  return fake.address()


def companies_info_generator(country_code, num_companies):
  companies = dict()
  num_companies_generated = 0

  while num_companies_generated != num_companies:
    description = company_generator(country_code)
    if description not in companies:
      if np.random.randint(0,2):
        address = address_generator(country_code)
      else:
        address = ""
      companies[description] = {"num_entry": 1, "address": address}
      num_companies_generated += 1

  return companies




def compute_entry_range():
  """ Compute the range(min, max) of the entity """
  new_range = []
  for i in range(MIN_RANGE_ENTRY,MAX_RANGE_ENTRY):
    if i < 5:
      new_range += [i for _ in range((MAX_RANGE_ENTRY) // 2)]
    elif i > 5 and i < 15:
      new_range += [i for _ in range(MAX_RANGE_ENTRY)]
    else:
        if(i % 10 == 0): new_range.append(i)

  return new_range



def data_generator(dataset):
  new_range = compute_entry_range()

  for i in tqdm(range(NUM_IBAN)):
    # generazione BIC
    bic, bic_country_code = bic_manual_generator()

    # generazione IBAN
    iban = iban_generator()

    # generazione numero di entry per questo IBAN
    num_iban_entry = np.random.choice(new_range)

    # scelta se IBAN è condiviso e, in caso, da quanti titolari
    is_shared = np.random.randint(0,2) if num_iban_entry != 1 else 0
    if is_shared:
      num_holders = np.random.randint(MIN_RANGE_HOLDERS, num_iban_entry+1 if num_iban_entry < MAX_RANGE_HOLDERS else MAX_RANGE_HOLDERS+1)
    else:
      num_holders = 1

    # generazione nome società e eventuali indirizzi
    country_code = np.random.choice(FAKER_COUNTRY_CODES)
    companies_info = companies_info_generator(country_code, num_companies=num_holders)

    # scelta quante entry per ogni società
    if is_shared and num_holders != num_iban_entry:
      entry_to_generate = num_iban_entry - num_holders
      while entry_to_generate != 0:
        random_holder = np.random.choice(list(companies_info.keys()))
        new_num_entry = np.random.randint(1,entry_to_generate+1) if entry_to_generate != 1 else 1
        companies_info[random_holder]["num_entry"] += new_num_entry
        entry_to_generate -= new_num_entry
    elif not is_shared:
      companies_info[list(companies_info.keys())[0]]["num_entry"] = num_iban_entry

    for name,info in companies_info.items():
      if info["num_entry"] != 1:
        aliases = generate_permutations(name, info["num_entry"], T, C, V, EDIT)
        for alias in aliases:
          dataset.loc[len(dataset.index)] = [bic, iban, bic_country_code, alias, companies_info[name]["address"], is_shared, name]
      else:
        dataset.loc[len(dataset.index)] = [bic, iban, bic_country_code, name, companies_info[name]["address"], is_shared, name]

  return dataset



def create_dataset():
  return pd.DataFrame(columns=["BIC", "AccountNumber", "CTRYbnk", "Name", "Address", "IsShared", "Holder"])



def save_dataset(dataset, filePath):
  """ Save the dataset generated """
  dataset.to_excel(filePath)



def get_dataset_filePath():
  """ return a new dataset name including actual datetime """
  now = datetime.now()
  return "./output/dataset_" + now.strftime("%d-%m-%Y_%H:%M:%S") + ".xlsx"



def main():
  emptyDataset = create_dataset()
  dataset = data_generator(emptyDataset)
  path = get_dataset_filePath()
  save_dataset(dataset, path)



if __name__ == "__main__":
  print("Check parameter...")
  check_parameters()
  print("Main...")
  main()
  print("Done...")
  print("Dataset saved in /output foldes.")