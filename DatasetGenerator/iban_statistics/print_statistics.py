import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mode
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.stats import skew, kurtosis



class SaveOutput():
  """ Simple class to save data on file """

  def __init__(self, filePath, fileName=None, printAll=False, debug=False):
    self.debug = debug
    self.printAll = printAll
    if not self.debug:
      if fileName: self.fileName = fileName
      else: self.fileName = "log_" + str(datetime.now()).split(".")[0].replace(" ", "_") + ".txt"
      self.filePath = filePath + self.fileName
      open(self.filePath, "w").close()

  def __call__(self, text = None):
    if not self.debug:
      with open(self.filePath, "a") as file:
        if text is None:
          file.write("\n")
        else:
          file.write(text)
          file.write("\n")
    if (self.printAll):
      print(text)


def main():
    """ """
    
    # INIT ------------------------------------------------------------------

    if len(sys.argv) < 2:
        print("Type a dataset path after the command script")
        exit()

    datasetPath = sys.argv[1]
    datasetName = os.path.basename(datasetPath).split(".")[0]
    if ".xlsx" in datasetPath:
        dataset = pd.read_excel(datasetPath)
    else: dataset = pd.read_csv(datasetPath)
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop("Unnamed: 0", axis=1)

    print_output = SaveOutput(filePath="./output_statistics/", fileName=datasetName + ".txt", printAll=True)
    
    # STATS  ------------------------------------------------------------------

    print_output("Dataset: " + datasetPath)
    print_output("\nPreview Dataset\n")
    print_output(dataset.head(2).to_markdown())
    print_output("\n")

    print_output("Dataset statistics\n\n")
    print_output(dataset.describe(include='all').to_markdown())
    print_output()
    print_output()
    
    print_output(str(dataset.columns))
    numberRow, numberCol = dataset.shape
    ibans = dataset.groupby(["AccountNumber"])

    print_output("Number row: " + str(numberRow))
    print_output("Number of unique IBAN: " + str(len(ibans)))
    count = dataset["AccountNumber"].value_counts().to_dict()
    
    figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 16})
    counts, bins, _ = plt.hist(dataset["AccountNumber"].value_counts().sort_values(), bins=range(min(count.values()), max(count.values())+1))
    
    plt.title('Frequenza transazioni')
    plt.xlabel('Dimensione gruppo Iban')
    plt.ylabel('Numero di gruppi')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.tight_layout()
    #plt.xticks(range(float(bins.min()),float(bins.max()),1.0))
    
    min_x = bins.min()
    max_x = bins.max()
    print_output("Min Number of transaction: " + str(min_x))
    print_output("Max Number of transaction: " + str(max_x))
    
    count = Counter(count.values())
    count = dict(sorted(count.items(), key=lambda item: item[1]))
    
    print_output("\n\n")
    print_output("STATS IBANS: ")
    print_output("Statistics on number of transaction: ")
    print_output("")
    print_output("Number of IBAN" + "\t\t Number of tranòsaction")
    for el in count:
        print_output("Group number iban: " + str(el) + " \ttransaction: " + str(count[el]))
    
    print_output("\n\n")
    print_output("Statistics on holder (ONLY SHARED)")
    ibans = dataset["AccountNumber"].unique()
    values = []
    for iban in ibans:
        names = dataset["Holder"].loc[dataset["AccountNumber"] == iban]
        names = list(set([el for el in names if not isinstance(el, float)]))
        values.append(len(names))

    values2 = [el for el in values if el != 1]
    object2 = {
        "min": np.min(values2),
        "max": np.max(values2),
        "median": np.median(values2),
        "mode": mode(values2),
        "mean": np.mean(values2),
        "std_dev":np.std(values2),
        "skewness":skew(values2),
        "kurt":kurtosis(values2)
    }

    print_output()
    print_output("Distribution on Holders per iban:")
    names = pd.DataFrame(object2)
    print_output(names.head(1).to_markdown())

    print_output()
    print_output("STATS HOLDER: ")
    count = Counter(values)
    count = dict(sorted(count.items(), key=lambda item: item[1]))
    print_output("Group holder number" + "\t\t Total number occurrences")
    for el in count:
        print_output("holder: " + str(el) + " \tNumber occurrence: " + str(count[el]))
        
    print_output("")
    plt.savefig("./output_statistics/image/" + datasetName + "_stats.png")
    plt.show()
    
    print_output("\n\n")
    print_output("Is Shared Stats:")
    groupped = dataset.groupby(["AccountNumber"])
    num_0 = 0
    num_1 = 0
    for i, data in groupped:
        if data['IsShared'].tolist()[0] == 0:num_0 += 1
        else: num_1 += 1
        
    print_output("Num Shared: " + str(num_1))
    print_output("Num Not Shared: " + str(num_0))    
    print_output("Number ibans: " + str(len(groupped)))

    

main()
