import pandas as pd
import sys
import json

PRINT_ERRORS = True



def print_result(num_errors):
  if num_errors == 0:
    print("\nDataset is correct.")
  else:
    print("\nDataset is not correct. Total errors: " + str(num_errors))


def test_parameters(dataset, parameters_file_path, num_errors):
  with open(parameters_file_path, "r") as data_file:
    parameters = json.load(data_file)
  
  param_num_iban = parameters["num_iban"]
  param_min_range_entry = parameters["min_range_entry"]
  param_max_range_entry = parameters["max_range_entry"]
  param_min_range_holders = parameters["min_range_holders"]
  param_max_range_holders = parameters["max_range_holders"]

  real_num_iban = 0
  num_min_entry = float("inf")
  num_max_entry = 0
  num_min_holders = float("inf")
  num_max_holders = 0
  for id, group in dataset.groupby(["AccountNumber"]):
    real_num_iban += 1

    num_entry = len(group)
    if num_entry < num_min_entry:
      num_min_entry = num_entry
    if num_entry > num_max_entry:
      num_max_entry = num_entry
    
    is_shared = set(group["IsShared"]).pop()
    if is_shared:
      num_holders = len(set(group["Holder"]))
      if num_holders < num_min_holders:
        num_min_holders = num_holders
      if num_holders > num_max_holders:
        num_max_holders = num_holders

  if real_num_iban != param_num_iban:
    num_errors += 1
    print("\nError "+str(num_errors)+": there are " + str(real_num_iban) + " IBANs but " + str(param_num_iban) + " were required.")
  if num_min_entry < param_min_range_entry:
    num_errors += 1
    print("\nError "+str(num_errors)+": the minimum entry number are " + str(num_min_entry) + " but " + str(param_min_range_entry) + " were required.")
  if num_max_entry > param_max_range_entry:
    num_errors += 1
    print("\nError "+str(num_errors)+": the maximum entry number are " + str(num_max_entry) + " but " + str(param_max_range_entry) + " were required.")
  if num_min_holders < param_min_range_holders:
    num_errors += 1
    print("\nError "+str(num_errors)+": the minimum holders number when account is shared are " + str(num_min_holders) + " but " + str(param_min_range_holders) + " were required.")
  if num_max_holders > param_max_range_holders:
    num_errors += 1
    print("\nError "+str(num_errors)+": the maximum holders number when account is shared are " + str(num_min_holders) + " but " + str(param_min_range_holders) + " were required.")
  
  return num_errors
  

def test_account_number(dataset, num_errors):
  ''' Checking for inconsistencies on account number. '''
  prev_account_number = ""
  list_account_number = []
  for index,row in dataset.iterrows():
    if prev_account_number == "":
      prev_account_number = row["AccountNumber"]
    if row["AccountNumber"] != prev_account_number:
      if prev_account_number in list_account_number:
        num_errors += 1
        if PRINT_ERRORS:
          print("\nError "+str(num_errors)+": account number "+prev_account_number+" is already used.")
      
      list_account_number.append(prev_account_number)
      prev_account_number = row["AccountNumber"]
  
  return num_errors


def test_sharing_and_address(dataset, num_errors):
  ''' Checking for inconsistencies on sharing and holders. '''

  for id, group in dataset.groupby(["AccountNumber","BIC"]):
    shared_values = set(group["IsShared"])
    if len(shared_values) > 1 :
      num_errors += 1
      if PRINT_ERRORS:
        print("\nError "+str(num_errors)+": entry with same account number not have same value of column \"IsShared\".")

    is_shared = group.iloc[0]["IsShared"]
    holders = set(group["Holder"])
    if is_shared:
      if len(holders) == 1:
        num_errors += 1
        if PRINT_ERRORS:
          print("\nError "+str(num_errors)+": iban is shared but there is only one holder.")
          print(group.to_string())
    else:
      if len(holders) > 1:
        num_errors += 1
        if PRINT_ERRORS:
          print("\nError "+str(num_errors)+": iban is not shared but there are multiple holders.")
          print(group.to_string())
  
  return num_errors


def test_empty_values(dataset, num_errors):
  name_columns = ["BIC","AccountNumber","CTRYbnk","Name","IsShared","Holder"]

  for name_column in name_columns:
    values = [str(elem).replace(" ","") for elem in list(dataset[name_column])]
    empty_values = values.count("") + values.count(None)
    if empty_values != 0:
      num_errors += 1
      print("\nError "+str(num_errors)+": there are " + empty_values + " empty "+name_column+".")

  return num_errors


def open_dataset(file_path):
  file_path = sys.argv[1]
  if file_path[-3:] == "csv":
    return pd.read_csv(file_path)
  else:
    return pd.read_excel(file_path, engine='openpyxl')


def main(dataset_file_path, parameters_file_path):
  dataset = open_dataset(dataset_file_path)
  
  num_errors = test_empty_values(dataset, num_errors=0)
  num_errors = test_account_number(dataset, num_errors)
  num_errors = test_sharing_and_address(dataset, num_errors=num_errors)
  
  if parameters_file_path is not None:
    num_errors = test_parameters(dataset, parameters_file_path, num_errors)
  
  print_result(num_errors)
  

if __name__ == "__main__":
  dataset_file_path = sys.argv[1]
  parameters_file_path = sys.argv[2] if len(sys.argv) > 2 else None

  main(dataset_file_path, parameters_file_path)