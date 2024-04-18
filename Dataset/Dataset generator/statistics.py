from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from scipy.stats import mode
import pandas as pd
import numpy as np
import sys
import os


def main():
    """ """
    
    # INIT ------------------------------------------------------------------

    if len(sys.argv) < 2:
        print("Type a dataset path after the command script")
        exit()

    datasetPath = sys.argv[1]
    datasetName = os.path.basename(datasetPath).split(".")[0]
    dataset = pd.read_excel(datasetPath, index_col=0)
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop("Unnamed: 0", axis=1)
    
    if "AccountNumber_anon" in dataset.columns: 
        columnName = "AccountNumber_anon"
        Name = "Name_anon"

    else: 
        columnName = "AccountNumber"
        Name = "Name"

    
    
    # STATS  ------------------------------------------------------------------

    print("Dataset: ", datasetPath)
    print("\nPreview Dataset\n")
    print(dataset.head(2).to_markdown())
    print("\n")

    print("Dataset statistics\n\n")
    print(dataset.describe(include='all').to_markdown())
    print()
    print()
    
    numberRow, numberCol = dataset.shape
    ibans = dataset.groupby([columnName])

    print("Number row: ", numberRow)
    print("Column Name: ", columnName)
    print("Number of unique IBAN: ", len(ibans))
    
    counts, bins, _ = plt.hist(dataset[columnName].value_counts())
    plt.title('Frequenza transazioni')
    plt.xlabel('Dimensione gruppo Iban')
    plt.ylabel('Numero di gruppi')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.tight_layout()
    
    min_x = bins.min()
    max_x = bins.max()
    print("Min Number of transaction: ", min_x)
    print("Max Number of transaction: ", max_x)
    print()
    print("statistics on names and address")
    ibans = dataset[columnName].unique()
    values = []
    for iban in ibans:
        names = dataset[Name].loc[dataset[columnName] == iban]
        names = [el for el in names if not isinstance(el, float)]
        if "Address" in dataset.columns:
            address = dataset["Address"].loc[dataset[columnName] == iban]
            address = [el for el in address if not isinstance(el, float)]
            values.append(len(names) + len(address))
        else:
            values.append(len(names))

    object2 = {
        "mean": np.mean(values),
        "median": np.median(values),
        "std_dev":np.std(values),
        "skewness":skew(values),
        "kurt":kurtosis(values)
    }

    print("Distribution on names and address per iban:")
    names = pd.DataFrame(object2,index=[0])
    print(names.to_markdown())


    plt.savefig("./output_statistics/image/distribuzione_iban_" + datasetName + ".png")
    plt.show()

    # WRITE ON FILE  ------------------------------------------------------------------

    f = open("./output_statistics/" + datasetName + "_statistics.txt", "w")
    f.write(" ---- Preview Dataset ---- \n\n")
    f.write(dataset.head(10).to_markdown())
    f.write("\n\n")

    f.write("\n\n ---- Dataset statistics ---- \n\n")
    f.write(dataset.describe(include='all').to_markdown())
    f.write("\n\n\n")

    f.write(" ---- Distribution on names and address per iban ---- \n\n")
    f.write(names.to_markdown())
    f.write("\n\n\n")

    f.write("Number row: " + str(numberRow) + "\n")
    f.write("Number of unique IBAN: " + str(len(ibans)) + "\n")
    f.write("Min Number of transaction: " + str(min_x) + "\n")
    f.write("Max Number of transaction: " + str(max_x) + "\n")

    f.close()


main()
