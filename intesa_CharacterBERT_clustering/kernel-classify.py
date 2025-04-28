import os
from typer import Typer
from rich.console import Console

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from lib.datasetManipulation import labeled_pairs
from lib.kernel_sim_data_utils import load_df, load_sim_data, save_sim_data
import lib.string_kernels as sk
from federated_learning import create_couple_df

# GLOBALS
app = Typer()
console = Console()

DF_TRAIN_PATH = "dataset/split_dataset/df_train.csv"
DF_TEST_PATH = "dataset/split_dataset/df_test.csv"
SIM_TRAIN_PATH = "dataset/similarity_train_seed_%d.csv"
SIM_TEST_PATH = "dataset/similarity_test_seed_%d.csv"

# COMMANDS

@app.command()
def create_dataset(seed: int, n_features: int = 7, overwrite: bool = False):
    sim_train_path = SIM_TRAIN_PATH % seed
    sim_test_path = SIM_TEST_PATH % seed

    train,test = load_df(
        train_path=DF_TRAIN_PATH,
        test_path=DF_TEST_PATH 
    )

    if os.path.exists(sim_train_path) and not overwrite:
        console.log(f"Train data already exists at {sim_train_path}")
    else:
        console.log("Saving train data")
        save_sim_data(sim_train_path, train, n_features, oversample=True)

    if os.path.exists(sim_test_path) and not overwrite:
        console.log(f"Test data already exists at {sim_test_path}")
    else:
        console.log("Saving test data")
        save_sim_data(sim_test_path, test, n_features)

@app.command()
def show_dataset(seed: int):
    train, test = load_sim_data(
        train_path=SIM_TRAIN_PATH % seed,
        test_path=SIM_TEST_PATH % seed
    )

    console.log(f"Train data:\n {train.head()}")
    console.log(f"Test data:\n {test.head()}")

@app.command()
def classify(seed: int):
    train, test = load_sim_data(
        train_path=SIM_TRAIN_PATH % seed,
        test_path=SIM_TEST_PATH % seed
    )

    scaler = MinMaxScaler()
    train.iloc[:, :-1] = scaler.fit_transform(train.iloc[:, :-1])
    test.iloc[:, :-1] = scaler.transform(test.iloc[:, :-1])


    lr = LogisticRegression(max_iter=1000)
    train_x = train.iloc[:, :-1].values
    train_y = train.iloc[:, -1].values
    test_x = test.iloc[:, :-1].values
    test_y = test.iloc[:, -1].values
    lr.fit(train_x, train_y)

    train_preds = lr.predict(train_x)
    test_preds = lr.predict(test_x)

    import wandb
    wandb.init(
        project="fl-ner",
        entity="mlgroup",
        name=f"kernel-lr-{seed}",
        group="Roberto-1",
        tags=["flner", "centralized", "spectrum-kernel"],
        config={
            "seed": seed,
            "train_size": len(train),
            "test_size": len(test),
            "kernel": "spectrum",
            "n_features": train.shape[1] - 1,
            "model": "LogisticRegression",
            "max_iter": 1000
        }
    )

    # create wandb table for the performance metrics
    table = wandb.Table(columns=["metric", "value"])
    cr_test = classification_report(test_y, test_preds, output_dict=True)
    cr_train = classification_report(train_y, lr.predict(train_x), output_dict=True)

    console.log(f"Train classification report:\n {cr_train}")
    console.log(f"Test classification report:\n {cr_test}")

    train_table = wandb.Table(columns=["metric", "value"])
    train_table.add_data("accuracy", accuracy_score(train_y, train_preds))
    train_table.add_data("macro_f1", cr_train["macro avg"]["f1-score"])
    train_table.add_data("precision", cr_train["macro avg"]["precision"])
    train_table.add_data("recall", cr_train["macro avg"]["recall"])
    train_table.add_data("support", cr_train["macro avg"]["support"])

    test_table = wandb.Table(columns=["metric", "value"])
    test_table.add_data("accuracy", accuracy_score(test_y, test_preds))
    test_table.add_data("macro_f1", cr_test["macro avg"]["f1-score"])
    test_table.add_data("precision", cr_test["macro avg"]["precision"])
    test_table.add_data("recall", cr_test["macro avg"]["recall"])
    test_table.add_data("support", cr_test["macro avg"]["support"])

    wandb.log({"train_metrics": train_table})
    wandb.log({"test_metrics": test_table})
    

    wandb.log({"train_accuracy": accuracy_score(train_y, train_preds)})
    wandb.log({"test_accuracy": accuracy_score(test_y, test_preds)})
    wandb.log({"train_macro_f1": cr_train["macro avg"]["f1-score"]})
    wandb.log({"test_macro_f1": cr_test["macro avg"]["f1-score"]})

if __name__ == "__main__":
    app()