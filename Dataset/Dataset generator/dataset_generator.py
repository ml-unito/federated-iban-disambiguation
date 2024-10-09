# install all necessary libraries
#import os
#try: import pipreqs
#except ImportError as error: os.system('pip install pipreqs')
#os.system("pip install -r ./requirements.txt")

import re
import sys
import json
import string
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from faker import Faker
from datetime import datetime
from IPython.display import display
from lib.permutations import generate_permutations



with open('./config/parameters.json', "r") as data_file:
  parameters = json.load(data_file)

PATH_OUTPUT_FILE = parameters["path_output_file"]

# numero di iban
NUM_IBAN = parameters["num_iban"]

PROB_SHARED_ACCOUNT = parameters["prob_shared_account"]
PROB_ADDRESS = parameters["prob_address"]

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
STATS_FROM_DATASETS = parameters["stats_from_dataset"]

if STATS_FROM_DATASETS:
  try:
    with open('./config/statistics.json', "r") as data_file:
      statistics = json.load(data_file)
      
    IBAN_VALUES = statistics["iban_values"]
    IBAN_PROBA = statistics["iban_proba"]
  except:
    print("Exception: can't read statistics.json file. It must be generated with extract_statistics.py")
    exit()



def check_parameters():
  if MAX_RANGE_ENTRY < MIN_RANGE_ENTRY:
    raise Exception("Exception: max_range_entry must be grater that min_range_entry.")
  if MIN_RANGE_ENTRY <= 0 or MAX_RANGE_ENTRY <=0:
    raise Exception("Exception: max_range_entry and min_range_entry must be positive number.")
  if MAX_RANGE_HOLDERS < MIN_RANGE_HOLDERS:
    raise Exception("Exception: max_range_holders must be grater that min_range_holders.")
  if MIN_RANGE_HOLDERS <= 0 or MAX_RANGE_HOLDERS <=0:
    raise Exception("Exception: max_range_holders and min_range_holders must be positive number.")
  if MIN_RANGE_HOLDERS < 2:
    raise Exception("Exception: min_range_holders must be grater than 1, because it is used when IBAN is shared.")
  
  # Check of probability values
  if PROB_SHARED_ACCOUNT > 1 or PROB_SHARED_ACCOUNT < 0:
    raise Exception("Exception: the probability of an account being shared must be a number between 0 and 1.")
  if PROB_ADDRESS > 1 or PROB_ADDRESS < 0:
    raise Exception("Exception: the probability of an address being generated must be a number between 0 and 1.")


def create_faker_objects():
  faker_objects = dict()
  for country_code in tqdm(FAKER_COUNTRY_CODES, desc="Faker objects generation"):
    faker_objects[country_code] = Faker(country_code)

  return faker_objects


def bic_manual_generator():
  country_code = random.choice(BIC_COUNTRY_CODES)
  bank_code = "".join([random.choice(string.ascii_uppercase) for _ in range(4)])
  location_code = random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase+'012')
  return bank_code + country_code + location_code, country_code


def iban_generator(faker_objects):
  country_code = random.choice(list(FAKER_COUNTRY_CODES.keys()))
  fake = faker_objects[country_code]
  return fake.unique.iban()


def company_generator(faker_objects, country_code):
  fake = faker_objects[country_code]
  return fake.company()


def address_generator(faker_objects, country_code):
  fake = faker_objects[country_code]
  address = fake.address()

  is_valid = 0
  while not is_valid:
    if country_code == "en_US" and ("APO" in address or "DPO" in address or "FPO" in address):
      address = fake.address()
    else:
      is_valid = 1

  return address


def companies_info_generator(faker_objects, country_code, num_companies):
  ''' Generates company names and their addresses, based on the country code
    and the number of companies specified in parameters.'''
  companies = dict()
  num_companies_generated = 0

  while num_companies_generated != num_companies:
    description = company_generator(faker_objects, country_code)
    if description not in companies:
      address = address_generator(faker_objects, country_code)
      companies[description] = {"num_entry": 1, "address": address}
      num_companies_generated += 1

  return companies


def generate_entry_number():
  """ Random generation of the number of entries to be included in the dataset for a specific iban """
  
  if not STATS_FROM_DATASETS:
    if MAX_RANGE_ENTRY > MIN_RANGE_ENTRY*3:
      type_entry_number = np.random.choice(["low","high"], p=[0.2,0.8])
      if type_entry_number == "low":
        num = np.random.randint(MIN_RANGE_ENTRY, MAX_RANGE_ENTRY//3 if MAX_RANGE_ENTRY//3 != 1 else 2)
      else:
        num = np.random.randint(MAX_RANGE_ENTRY//3, MAX_RANGE_ENTRY+1)
      return num
    else:
      return np.random.randint(MIN_RANGE_ENTRY, MAX_RANGE_ENTRY+1)
  else:
    return np.random.choice(IBAN_VALUES, p=IBAN_PROBA)


def generate_holder_number(num_entries):
  max_low = 4
  max_middle = 10
  new_max_holders = MAX_RANGE_HOLDERS if MAX_RANGE_HOLDERS < num_entries else num_entries

  if new_max_holders <= max_low+1 or MIN_RANGE_HOLDERS > max_middle or (new_max_holders <= max_middle and MIN_RANGE_HOLDERS > max_low):
    num_holders = np.random.randint(MIN_RANGE_HOLDERS, new_max_holders+1)
    return num_holders
  else:
    list_type = ["low","middle","high"]
    if new_max_holders <= max_middle:
      list_type.remove("high")
    elif MIN_RANGE_HOLDERS > max_low:
      list_type.remove("low")

    prob = [0.6,0.3,0.1] if len(list_type) > 2 else [0.7,0.3]
    type_co_headings = np.random.choice(list_type, p=prob)

    if type_co_headings == "low":
      num_holders = np.random.randint(
        MIN_RANGE_HOLDERS, 
        max_low+1 if new_max_holders > max_low else new_max_holders+1
      )
    elif type_co_headings == "middle":
      num_holders = np.random.randint(
        max_low+1 if new_max_holders >= max_low+1 and MIN_RANGE_HOLDERS <= max_low+1 else MIN_RANGE_HOLDERS,
        max_middle+1 if new_max_holders > max_middle else new_max_holders
      )
    else:    
      num_holders = np.random.randint(
        max_middle+1 if new_max_holders >= max_middle+1 and MIN_RANGE_HOLDERS <= max_middle+1 else MIN_RANGE_HOLDERS,
        new_max_holders+1
      )
    
    return num_holders


def get_address_number(info_address, country_code):
  ''' It returns the address number and/or more information about that from the
    address if it is present, otherwise the empty string. '''
  index_number = FAKER_COUNTRY_CODES[country_code]["pos_elem"]["number"]
  number = info_address[index_number] if index_number != 1 else None

  index_extra_info = FAKER_COUNTRY_CODES[country_code]["pos_elem"]["extra_info"]
  extra_info = info_address[index_extra_info] if index_extra_info != 1 else None

  if number is not None and extra_info is not None:
    return number + " " + extra_info
  elif number is not None:
    return number
  elif extra_info is not None:
    return extra_info
  else:
    return ""


def get_address_street(info_address, country_code):
  ''' It returns the address street if it is present, otherwise the empty string. '''
  street = info_address[FAKER_COUNTRY_CODES[country_code]["pos_elem"]["street"]]
  return street if street is not None else ""


def get_address_city(info_address, country_code):
  ''' It returns the address city if it is present, otherwise the empty string. '''
  city = info_address[FAKER_COUNTRY_CODES[country_code]["pos_elem"]["city"]]
  return city if city is not None else ""


def get_address_postal_code(info_address, country_code):
  ''' It returns the address postal code if it is present, otherwise the empty string. '''
  postal_code = info_address[FAKER_COUNTRY_CODES[country_code]["pos_elem"]["postal_code"]]
  return postal_code if postal_code is not None else ""


def get_address_state(info_address, country_code):
  ''' It returns the address state if it is present, otherwise the empty string. '''
  index = FAKER_COUNTRY_CODES[country_code]["pos_elem"]["state"]
  state = info_address[index] if index != -1 else None
  return state if state is not None else ""


def get_country(country_code):
  ''' It returns the name of the country from country code. '''
  return FAKER_COUNTRY_CODES[country_code]["country"]


def change_address_format(address, country_code):
  ''' Randomly it changes the address format and possibly introduces write errors. '''

  # With regular expression, it extracts all informations from the address.
  regex = re.compile(FAKER_COUNTRY_CODES[country_code]["regex"])
  info_address = re.split(regex,address)[1:-1]

  # Choose randomly the type of new format of the address.
  action = np.random.choice([
    "symbols", "only_city", "only_country", "city_and_country", 
    "city_and_short_country", "postal_code_and_city",
    "format1","format2","format3","format4",
    "original_format"
    ]) #p=[0.05,0.15,0.1,0.1,0.1,0.05,0.15,0.15,0.15])
  only_symbols = False
  new_address_elems = []

  if action == "symbols":
    only_symbols = True
    symbols = [".","-","_","X","&","/","*","#"]
    index_symbol = np.random.randint(0,len(symbols))
    new_address_elems = [symbols[index_symbol] for i in range(np.random.randint(1,5))]
  elif action == "only_city":
    new_address_elems = [get_address_city(info_address, country_code)]
  elif action == "only_country":
    new_address_elems = [get_country(country_code)]
  elif action == "city_and_country" or action == "city_and_short_country":
    new_address_elems = [get_address_city(info_address, country_code) + " " + (country_code[3:] if action == "city_and_short_country" else get_country(country_code))]
  elif action == "postal_code_and_city":
    postal_code = get_address_postal_code(info_address, country_code)
    city = get_address_city(info_address, country_code)      
    new_address_elems = [postal_code, city]
  elif action == "format1":
    street = get_address_street(info_address, country_code)
    number = get_address_number(info_address, country_code) 
    city = get_address_city(info_address, country_code)
    country = get_country(country_code)
    new_address_elems = [street, number, city, country]
  elif action == "format2" or action == "format3":
    street = get_address_street(info_address, country_code)
    number = get_address_number(info_address, country_code)
    postal_code = get_address_postal_code(info_address, country_code)
    city = get_address_city(info_address, country_code)
    state = get_address_state(info_address, country_code)
    if state != "" and np.random.randint(0,2):
      state = "(" + state + ")"
    country = country_code[3:] if action == "format2" else get_country(country_code)
    new_address_elems = [street, number, city, state, country]
  elif action == "format4":
    number = get_address_number(info_address, country_code)
    street = get_address_street(info_address, country_code)
    city = get_address_city(info_address, country_code)
    new_address_elems = [number, street, city]
  elif action == "original_format":
    new_address_elems = [info if info is not None else "" for info in info_address]

  # Random introduction of write error in the address, such as removing
  # characters, introducing symbols, or adding/removing spaces between words.
  address = ""
  for index_elem, elem in enumerate(new_address_elems):
    if elem != "":
      if not only_symbols and not elem.isnumeric() and len(elem)!=2 and np.random.choice([0,1], p=[0.90,0.10]):
        index_char = np.random.randint(0,len(elem))
        address += elem[0:index_char]+elem[index_char+1:]
      else:
        address += elem
      
      if index_elem != len(new_address_elems)-1:
        space_action = np.random.choice(
          ["no_space","add_more_space","replace_space_with_symbol","one_space"], 
          p=[0.1,0.20,0.30,0.40]
        )
        if space_action == "add_more_space":
          address += "  "
        elif not only_symbols and space_action == "replace_space_with_symbol":
          symbols = ["-",",","/"]
          symbol = np.random.choice(symbols)
          address += symbol
        elif space_action == "one_space":
          address += " "
  address = address[:len(address)-2] if address[len(address)-1] == " " else address
  
  if "'" in address and np.random.randint(0,2):
    address = address.replace("'","")
  
  return address


def data_generator(dataset, faker_objects):
  for i in tqdm(range(NUM_IBAN),desc="Dataset generation"):
    # Generation of BIC code
    bic, bic_country_code = bic_manual_generator()

    # Generation of IBAN code and the number of related entries. Choice of whether it is a
    # shared account and the number of associated holders.
    iban = iban_generator(faker_objects)
    num_iban_entry = generate_entry_number()
    is_shared = np.random.choice([0,1], p=[1-PROB_SHARED_ACCOUNT, PROB_SHARED_ACCOUNT]) if num_iban_entry != 1 else 0
    if is_shared:
      if num_iban_entry < MIN_RANGE_HOLDERS:
        num_iban_entry = MIN_RANGE_HOLDERS
      num_holders = generate_holder_number(num_iban_entry)
    else:
      num_holders = 1

    # Generation of company names and any related addresses.
    country_code = np.random.choice(list(FAKER_COUNTRY_CODES.keys()))
    companies_info = companies_info_generator(faker_objects, country_code, num_companies=num_holders)
    
    # Choice number of entries for each company.
    if is_shared and num_holders != num_iban_entry:
      entry_to_generate = num_iban_entry - num_holders
      while entry_to_generate != 0:
        random_holder = np.random.choice(list(companies_info.keys()))
        new_num_entry = np.random.randint(1,entry_to_generate+1) if entry_to_generate != 1 else 1
        companies_info[random_holder]["num_entry"] += new_num_entry
        entry_to_generate -= new_num_entry
    elif not is_shared:
      companies_info[list(companies_info.keys())[0]]["num_entry"] = num_iban_entry

    # Adding data generated in the dataset, possibly permuting company names
    # and addresses.
    iban_dataframe = create_dataset()
    size_iban_dataframe = 0
    for name,info in companies_info.items():
      if info["num_entry"] != 1:
        aliases = generate_permutations(name, info["num_entry"], T, C, V, EDIT)
        for alias in aliases:
          new_address = change_address_format(companies_info[name]["address"], country_code) if np.random.choice([0,1], p=[1-PROB_ADDRESS,PROB_ADDRESS]) else ""
          iban_dataframe.loc[size_iban_dataframe] = [bic, iban, bic_country_code, alias, new_address, is_shared, name]
          size_iban_dataframe += 1
      else:
        address = change_address_format(companies_info[name]["address"], country_code) if np.random.choice([0,1], p=[1-PROB_ADDRESS,PROB_ADDRESS]) else ""
        iban_dataframe.loc[size_iban_dataframe] = [bic, iban, bic_country_code, name, address, is_shared, name]
        size_iban_dataframe += 1
    
    dataset = pd.concat([dataset, iban_dataframe], ignore_index=True)

  return dataset


def create_dataset():
  return pd.DataFrame(columns=["BIC", "AccountNumber", "CTRYbnk", "Name", "Address", "IsShared", "Holder"])


def save_dataset(dataset, filePath):
  """ Save the dataset generated """
  dataset.to_csv(filePath)


def get_dataset_filePath():
  """ return a new dataset name including actual datetime """
  now = datetime.now()
  return "./output/dataset_" + now.strftime("%d-%m-%Y_%H-%M-%S") + ".csv"


def print_dataset(dataset, maxLine = 20):
  """ Print first maxLine row of the datset """
  dataset = dataset.iloc[:maxLine,:]
  dataset = dataset[["AccountNumber", "Name", "Address", "IsShared"]]
  
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)
  pd.set_option('display.colheader_justify', 'center')
  pd.set_option('display.precision', 3)
  print("\n\n")
  display(dataset)
  print("\n\n")


def save_used_parameters(dataset_file_name):
  param = {
    "dataset_file_name": dataset_file_name,
    "num_iban": NUM_IBAN,
    "min_range_entry": MIN_RANGE_ENTRY,
    "max_range_entry": MAX_RANGE_ENTRY,
    "min_range_holders": MIN_RANGE_HOLDERS,
    "max_range_holders": MAX_RANGE_HOLDERS,
    "prob_shared_account": PROB_SHARED_ACCOUNT,
    "prob_address": PROB_ADDRESS,
  }

  json_object = json.dumps(param, indent=2)

  with open(dataset_file_name[:-4] + "__parameters" + ".json", "w") as outfile:
    outfile.write(json_object)


def main():
  empty_dataset = create_dataset()
  faker_objects = create_faker_objects()

  dataset = data_generator(empty_dataset, faker_objects)

  path = get_dataset_filePath()
  save_dataset(dataset, path)
  save_used_parameters(path)

  if len(sys.argv) > 1 and sys.argv[1] == "show":
    print_dataset(dataset)


if __name__ == "__main__":
  print("\nTYPE: \t 'python dataset_generator.py show'  for a preview of the dataset .........\n")
  print("Check parameter...")
  check_parameters()
  print("Main...")
  main()
  print("Done...")
  print("Dataset and parameters used are saved in /output folders.")