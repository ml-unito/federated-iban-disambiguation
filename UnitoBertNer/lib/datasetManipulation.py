import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def load_dataset_excel(path):
    """ Load the dataset """
    
    dataset = pd.read_excel(path)
    if "Unnamed: 0" in dataset.columns: dataset = dataset.drop("Unnamed: 0", axis=1)
    return dataset


def load_dataset_csv(path):
    """ Load the dataset """
    
    dataset = pd.read_csv(path)
    if "Unnamed: 0" in dataset.columns: dataset = dataset.drop("Unnamed: 0", axis=1)
    return dataset


def save_dataset_csv(dataset, path, mode="w", index=False, header=True):
    """ Save the dataset """
    dataset.to_csv(path, mode=mode, index=index, header=header)


def save_dataset_excel(dataset, path):
    """ Save the dataset """
    dataset.to_excel(path)


def group_dataset(dataset, groupLen=10, placeholder="[NA]"):
    """ Group the dataset """

    grouped = dataset.groupby('AccountNumber').agg({
        'Name': lambda x: x.tolist()[:groupLen],
        'Address': lambda x: x.tolist()[:groupLen],
        'IsShared':lambda x: x.tolist()[0]
    }).reset_index()

    # Splitting the lists of names and addresses into separate columns
    names_df = grouped['Name'].apply(pd.Series).add_prefix('Name_')
    addresses_df = grouped['Address'].apply(pd.Series).add_prefix('Address_')

    # Merging the separated columns with the original DataFrame
    # adding account number and IsShared
    # filling missing values with placeholder
    expanded_df = pd.merge(grouped['AccountNumber'],  names_df, left_index=True, right_index=True)
    expanded_df = pd.merge(expanded_df, addresses_df, left_index=True, right_index=True)
    expanded_df = pd.merge(expanded_df, grouped['IsShared'], left_index=True, right_index=True)
    expanded_df.fillna(placeholder, inplace=True)

    return expanded_df




def old_group_dataset(dataset, groupLen=20, placeholder="[NA]"):
    """ Group the dataset """

    group = dataset.groupby(['AccountNumber'])
    newDF = {"AccountNumber":"", "IsShared":-1, "Holder": ""}

    for i in range(groupLen):
        newName = "field_" + str(i)
        newDF[newName] = pd.NaT

    df = pd.DataFrame()
    for g in group:
        for field in newDF:newDF[field] = placeholder

        newDF['AccountNumber'] = g[0][0]
        newDF['IsShared'] = list(g[1]['IsShared'])[0]
        newDF['Holder'] = list(g[1]['Holder'])[0]

        names = list(g[1]['Name'])
        names = [el for el in names if not isinstance(el, float)]
        address = list(g[1]['Address'])
        address = [el for el in address if not isinstance(el, float)]

        actual = 0
        for i in range(len(names)):
            if i < groupLen:
                newName = "field_" + str(i)
                newDF[newName] = names[i]
                actual += 1

        for i in range(len(address)):
            if actual < groupLen:
                newName = "field_" + str(actual)
                newDF[newName] = address[i]
                actual += 1

        df1 = pd.DataFrame(newDF, index=[0])
        df = pd.concat([df, df1], ignore_index=True)

    return df



def print_stats(dataset):
    """ Print the stats of the dataset """

    dataset = dataset.drop("BIC",axis=1)
    dataset = dataset.drop("CTRYbnk",axis=1)
    ibans = dataset["AccountNumber"].unique()

    values = []
    for iban in ibans:
        names = dataset["Name"].loc[dataset["AccountNumber"] == iban]
        names = [el for el in names if not isinstance(el, float)]
        address = dataset["Address"].loc[dataset["AccountNumber"] == iban]
        address = [el for el in address if not isinstance(el, float)]
        values.append(len(names) + len(address))
        

    object2 = {
        "min": np.min(values),
        "max": np.max(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std_dev":np.std(values),
        "skewness":skew(values),
        "kurt":kurtosis(values)
    }


    print("Dataset infos:\n")
    print(dataset.info())
    print("\n")
    print("Distribution of names and addresses per iban:")
    names = pd.DataFrame(object2,index=[0])
    print(names.to_markdown())
    print()

