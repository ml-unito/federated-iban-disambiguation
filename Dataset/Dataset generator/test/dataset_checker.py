import pandas as pd
import sys

PRINT_ERRORS = True



def print_result(num_errors):
  if num_errors == 0:
    print("\nDataset is correct.")
  else:
    print("\nDataset is not correct. Total errors: " + str(num_errors))


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


def open_dataset(file_path):
  file_path = sys.argv[1]
  if file_path[-3:] == "csv":
    return pd.read_csv(file_path)
  else:
    return pd.read_excel(file_path, engine='openpyxl')


def main(file_path):
  dataset = open_dataset(file_path)
  
  num_errors = test_account_number(dataset, num_errors=0)
  num_errors = test_sharing_and_address(dataset, num_errors=num_errors)

  print_result(num_errors)
  

if __name__ == "__main__":
  file_path = sys.argv[1]
  main(file_path)