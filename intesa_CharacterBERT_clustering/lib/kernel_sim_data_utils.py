import os
import pandas as pd
from lib.datasetManipulation import labeled_pairs
from rich.console import Console
from lib import string_kernels as sk

console = Console()


def load_df(train_path: str, test_path: str):
    cp_train_df = pd.read_csv(train_path)
    cp_test_df = pd.read_csv(test_path)

    train_pairs, train_labels = labeled_pairs(cp_train_df)
    test_pairs, test_labels = labeled_pairs(cp_test_df)

    train_xy = list(zip(train_pairs, train_labels))
    test_xy = list(zip(test_pairs, test_labels))

    # Create a dataframe from the train and test data, split the pairs into two columns
    train_xy = [[pair[0], pair[1], label] for pair, label in train_xy]
    train_df = pd.DataFrame(train_xy, columns=["str1", "str2", "label"])

    test_xy = [[pair[0], pair[1], label] for pair, label in test_xy]
    test_df = pd.DataFrame(test_xy, columns=["str1", "str2", "label"])

    return train_df, test_df

def kernel_features_from_pairs(s1, s2, n_features=4):
    return [
        sk.spectrum_kernel([s1], [s2], p=i)[0].item() for i in range(1, n_features + 1)
    ]

def save_sim_data(fname:str, data:pd.DataFrame, n_features, oversample:bool=False):
    sims = [kernel_features_from_pairs(s1, s2, n_features=n_features) + [label] 
                for s1, s2, label in data.itertuples(index=False)]
    
    if oversample:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X = [s[:n_features] for s in sims]
        y = [s[n_features] for s in sims]

        console.log(f"Oversampling data")
        X_res, y_res = sm.fit_resample(X, y)
        sims = [[*s[:n_features], label] for s, label in zip(X_res, y_res)]

    with open(fname, "w") as f:
        f.write(f"{','.join([f'p{i}' for i in range(1,n_features+1)])},label\n")
        for sim in sims:
            f.write(f",".join([str(s) for s in sim]) + "\n")

def load_sim_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df