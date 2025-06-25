import string
import pandas as pd
import regex as re
from typer import Typer
from collections import Counter
from unidecode import unidecode

app = Typer()

STOP_WORDS = ["CREDIT UNION", "CU", "SAVINGS", "VAT NUMBER", "VAT", "BILLS", "ACCOUNT", "BIC"]

REGEX_IBAN = r'^((NO)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{3}|(NO)[0-9A-Z]{13}|(BE)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}|(BE)[0-9A-Z]{14}|(DK|FO|FI|GL|NL)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{2}|(DK|FO|FI|GL|NL)[0-9A-Z]{16}|(MK|SI)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{3}|(MK|SI)[0-9A-Z]{17}|(BA|EE|KZ|LT|LU|AT)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}|(BA|EE|KZ|LT|LU|AT)[0-9A-Z]{18}|(HR|LI|LV|CH)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{1}|(HR|LI|LV|CH)[0-9A-Z]{19}|(BG|DE|IE|ME|RS|GB)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{2}|(BG|DE|IE|ME|RS|GB)[0-9A-Z]{20}|(GI|IL)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{3}|(GI|IL)[0-9A-Z]{21}|(AD|CZ|SA|RO|SK|ES|SE|TN)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}|(AD|CZ|SA|RO|SK|ES|SE|TN)[0-9A-Z]{22}|(PT)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{1}|(PT)[0-9A-Z]{23}|(IS|TR)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{2}|(IS|TR)[0-9A-Z]{24}|(FR|GR|IT|MC|SM)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{3}|(FR|GR|IT|MC|SM)[0-9A-Z]{25}|(AL|CY|HU|LB|PL)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}|(AL|CY|HU|LB|PL)[0-9A-Z]{26}|(MU)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{2}|(MU)[0-9A-Z]{28}|(MT)[0-9A-Z]{2}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{4}[ ][0-9A-Z]{3}|(MT)[0-9A-Z]{29})$'
REGEX_SWIFT = r'^([a-zA-Z]{4})(-+)((AF|AX|AL|DZ|AS|AD|AO|AI|AQ|AG|AR|AM|AW|AU|AT|AZ|BS|BH|BD|BB|BY|BE|BZ|BJ|BM|BT|BO|BQ|BA|BW|BV|BR|IO|BN|BG|BF|BI|CV|KH|CM|CA|KY|CF|TD|CL|CN|CX|CC|CO|KM|CG|CD|CK|CR|CI|HR|CU|CW|CY|CZ|DK|DJ|DM|DO|EC|EG|SV|GQ|ER|EE|ET|FK|FO|FJ|FI|FR|GF|PF|TF|GA|GM|GE|DE|GH|GI|GR|GL|GD|GP|GU|GT|GG|GN|GW|GY|HT|HM|VA|HN|HK|HU|IS|IN|ID|IR|IQ|IE|IM|IL|IT|JM|JP|JE|JO|KZ|KE|KI|KP|KR|KW|KG|LA|LV|LB|LS|LR|LY|LI|LT|LU|MO|MK|MG|MW|MY|MV|ML|MT|MH|MQ|MR|MU|YT|MX|FM|MD|MC|MN|ME|MS|MA|MZ|MM|NA|NR|NP|NL|NC|NZ|NI|NE|NG|NU|NF|MP|NO|OM|PK|PW|PS|PA|PG|PY|PE|PH|PN|PL|PT|PR|QA|RE|RO|RU|RW|BL|SH|KN|LC|MF|PM|VC|WS|SM|ST|SA|SN|RS|SC|SL|SG|SX|SK|SI|SB|SO|ZA|GS|SS|ES|LK|SD|SR|SJ|SZ|SE|CH|SY|TW|TJ|TZ|TH|TL|TG|TK|TO|TT|TN|TR|TM|TC|TV|UG|UA|AE|GB|US|UM|UY|UZ|VU|VE|VN|VG|VI|WF|EH|YE|ZM|ZW){1})(-+)(\w{2})$'
REGEX_BIC = r'^(([a-zA-Z]{4})(-+)([a-zA-Z]{2})(-+)(\w{2})(-+)(\w{3}))|(([a-zA-Z]{4})([-]{0,1})([a-zA-Z]{2})([-]{0,1})(\w{2})([-]{0,1})(\d{3}))$'
REGEX_NUMBERS = r'^[\d\s\W]+$'
REGEX_SYMBOLS = r"^\W*$"
REGEX_NUMBERS_BLOCK = r'(\d{5,})'

def generate_stop_words_regex() -> str:
  regex = r""
  for stop_word in STOP_WORDS:
    words = stop_word.split()
    if len(words) > 1:
      regex += "("
      for word in words:
        regex += r"(" + word + r")" + r"(\s{0,1})"
      regex = regex[:-9] + r")|"
    elif len(words) == 1:
      word = words[0]
      regex += r"((\s){1}|^)("+ word +r")((\s){1}|$)" + "|"

  regex = regex[:-1]
  
  return regex

REGEX_STOP_WORDS = generate_stop_words_regex()



def remove_unnecessary_punctuation(text: str) -> str:
  for punctuation in string.punctuation:
    text = text.replace(punctuation + punctuation, " ")
  return text


def remove_multiple_spaces(text: str) -> str:
	return re.sub(re.compile(r'\s+'), ' ', text)


def assign_most_common_name(names: str) -> str:
  return Counter(names).most_common(1)[0][0]


def remove_numbers_blocks(text: str) -> str:
  if re.search(REGEX_NUMBERS_BLOCK, text):
    res = re.findall(REGEX_NUMBERS_BLOCK, text)
    for elem in res:
      text = text.replace(elem,"")
    text = remove_multiple_spaces(text)
    text = remove_unnecessary_punctuation(text)
    text = text.strip()
  
  return text


def dataset_preprocessing(dataset: pd.DataFrame, name_log: str) -> pd.DataFrame:
  log = open("./out/"+name_log+".txt", "w")
  log.write("NAMES CHANGED IN DATASET\n\n\n")

  dataset = dataset.reset_index()
  dataset.fillna(0, inplace=True)
  dataset["OldName"] = dataset["Name"]

  for iban, group in dataset.groupby("AccountNumber", sort=False):
    indexs_to_change = []
    
    for index, row in group.iterrows(): 
      new_name = unidecode(row["Name"])     
      new_name = remove_multiple_spaces(new_name)
      new_name = remove_unnecessary_punctuation(new_name)
      new_name = new_name.strip()

      if re.match(REGEX_NUMBERS, new_name)\
        or re.match(REGEX_SYMBOLS, new_name)\
          or re.match(REGEX_IBAN, new_name)\
            or re.match(REGEX_SWIFT, new_name)\
              or re.match(REGEX_BIC, new_name)\
                or re.search(REGEX_STOP_WORDS, new_name):
        indexs_to_change.append(row["index"])
      else:
        # Check if name contains block of numbers
        new_name = remove_numbers_blocks(new_name)

        # Save original name or name edited in the column "Name"
        dataset.loc[row["index"], "Name"] = new_name
    
    if len(indexs_to_change):
      filter_dataset = dataset[~dataset.index.isin(indexs_to_change)]
      names = list(filter_dataset.loc[(filter_dataset["AccountNumber"] == iban)]["Name"])
      log.write(str(names) + "\n")
      # Assign most common name of group at allo rows in indexs_to_change
      # only if not all row  is to change.
      if len(group) != len(indexs_to_change):
        common_name = assign_most_common_name(names)
        dataset.loc[indexs_to_change, "Name"] = common_name
        log.write(str(dataset.loc[indexs_to_change, ["OldName","Name","Holder"]]) + "\n" + "-"*80 + "\n")
      else:
        # Caso in cui tutte le righe del gruppo sono casi speciali. Quindi vengono lasciati i nomi invariati.
        log.write("\n\nsame names\t" + str(indexs_to_change)+"\n"+ "-"*80 +"\n")

  log.close()
  return dataset


@app.command()
def split_dataset():
  # dataset = pd.read_csv("./dataset/benchmark_intesa.csv")
  # new_datasets = dataset_preprocessing(dataset=dataset)
  # new_datasets.to_csv("./dataset/benchmark_intesa_preprocessed.csv")

  df_client1 = pd.read_csv("./dataset/split_dataset/client1_train.csv").drop(columns=["Unnamed: 0"])
  df_client2 = pd.read_csv("./dataset/split_dataset/client2_train.csv").drop(columns=["Unnamed: 0"])
  df_client3 = pd.read_csv("./dataset/split_dataset/client3_train.csv").drop(columns=["Unnamed: 0"])
  df_client4 = pd.read_csv("./dataset/split_dataset/client4_train.csv").drop(columns=["Unnamed: 0"])
  df_test = pd.read_csv("./dataset/split_dataset/df_test.csv").drop(columns=["Unnamed: 0"])
  df_train = pd.read_csv("./dataset/split_dataset/df_train.csv").drop(columns=["Unnamed: 0"])
  
  df_client1_new = dataset_preprocessing(dataset=df_client1, name_log="log_df_client1")
  df_client1_new.to_csv("./dataset/split_dataset/client1_train_pp.csv")

  df_client2_new = dataset_preprocessing(dataset=df_client2, name_log="log_df_client2")
  df_client2_new.to_csv("./dataset/split_dataset/client2_train_pp.csv")

  df_client3_new = dataset_preprocessing(dataset=df_client3, name_log="log_df_client3")
  df_client3_new.to_csv("./dataset/split_dataset/client3_train_pp.csv")

  df_client4_new = dataset_preprocessing(dataset=df_client4, name_log="log_df_client4")
  df_client4_new.to_csv("./dataset/split_dataset/client4_train_pp.csv")

  df_test_new = dataset_preprocessing(dataset=df_test, name_log="log_df_test")
  df_test_new.to_csv("./dataset/split_dataset/df_test_pp.csv")

  df_train_new = dataset_preprocessing(dataset=df_train, name_log="log_df_train")
  df_train_new.to_csv("./dataset/split_dataset/df_train_pp.csv")


@app.command()
def dataset(path:str):
  df = pd.read_csv(path).drop(columns=["Unnamed: 0"])
  df_new = dataset_preprocessing(dataset=df, name_log="log_df_preproc")
  df_new.to_csv(path[:-4]+"_pp.csv")


if __name__ == "__main__":
  app()
