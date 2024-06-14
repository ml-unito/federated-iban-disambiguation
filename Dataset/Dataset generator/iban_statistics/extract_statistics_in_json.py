from collections import Counter
import pandas as pd
import numpy as np
import json
import sys


def main():
    """ """
    
    # INIT ------------------------------------------------------------------

    if len(sys.argv) < 2:
        print("\nType a dataset, first!")
        print("USAGE: python extract_statistics.py DATASET_PATH")
        print("where, DATASET_PATH is a .csv or .xlsx file")
        exit()

    datasetPath = sys.argv[1]
    if ".csv" in datasetPath: dataset = pd.read_csv(datasetPath)
    else: dataset = pd.read_excel(datasetPath)

    
    # STATS  ------------------------------------------------------------------

    print("Dataset Name: ", datasetPath)
    print("\n\nPreview Dataset\n")
    print(dataset.head(2).to_markdown())
    print("\n")

    print("Dataset statistics\n")
    print(dataset.describe(include='all').to_markdown())
    print("\n")


    count = dataset["AccountNumber"].value_counts().to_dict()
    count = Counter(count.values())
    count = dict(sorted(count.items(), key=lambda item: item[1]))
    number_iban = sum(count.values())
    prob = []
    for el in count:prob.append((el, count[el]/number_iban))
    values = [el[0] for el in prob]
    proba = [el[1] for el in prob]
    
    json_file = {
        "iban_values": values,
        "iban_proba": proba
    }

    json.dump(json_file, open("../config/statistics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

main()