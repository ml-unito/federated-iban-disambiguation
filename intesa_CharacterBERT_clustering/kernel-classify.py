import os
from typer import Typer
from rich.console import Console
from rich.progress import track

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from lib.datasetManipulation import labeled_pairs
from lib.kernel_sim_data_utils import load_df, load_sim_data, save_sim_data
import lib.string_kernels as sk
from federated_learning import create_couple_df

from lib.mlp import MLP
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import wandb

# GLOBALS
app = Typer()
console = Console()

DF_TRAIN_PATH = "dataset/split_dataset/df_train.csv"
DF_TEST_PATH = "dataset/split_dataset/df_test.csv"
SIM_TRAIN_PATH = "dataset/similarity_train_seed_%d%s.csv"
SIM_TEST_PATH = "dataset/similarity_test_seed_%d%s.csv"

def log_metrics(step, model, train_x, train_y, test_x, test_y, running_loss, print_classification_report=False):
    with torch.no_grad():
        train_preds = model(train_x.to("cuda:0")).argmax(dim=1).cpu().numpy()
        test_preds = model(test_x.to("cuda:0")).argmax(dim=1).cpu().numpy()

        cr_train = classification_report(train_y.numpy(), train_preds, output_dict=True)
        cr_train_str = classification_report(train_y.numpy(), train_preds, output_dict=False)
        cr_test = classification_report(test_y.numpy(), test_preds, output_dict=True)
        cr_test_str = classification_report(test_y.numpy(), test_preds, output_dict=False)
        
        train_accuracy = cr_train["accuracy"]
        test_accuracy = cr_test["accuracy"]
        train_f1 = cr_train["macro avg"]["f1-score"]
        test_f1 = cr_test["macro avg"]["f1-score"]
        f1_train_label_1 = cr_train["1"]["f1-score"]
        f1_train_label_0 = cr_train["0"]["f1-score"]
        f1_test_label_1 = cr_test["1"]["f1-score"]
        f1_test_label_0 = cr_test["0"]["f1-score"]
        precizione_train_label_1 = cr_train["1"]["precision"]
        precision_train_label_0 = cr_train["0"]["precision"]
        precision_test_label_1 = cr_test["1"]["precision"]
        precision_test_label_0 = cr_test["0"]["precision"]
        recall_train_label_1 = cr_train["1"]["recall"]
        recall_train_label_0 = cr_train["0"]["recall"]
        recall_test_label_1 = cr_test["1"]["recall"]
        recall_test_label_0 = cr_test["0"]["recall"]

        if print_classification_report:
            console.print("Train classification report")
            console.print(cr_train_str)
            console.print("Test classification report")
            console.print(cr_test_str)

        wandb.log(step =step, 
                    data={
            "train_loss": running_loss / step,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_macro_f1": train_f1,
            "test_macro_f1": test_f1,
            "train_f1_label_1": f1_train_label_1,
            "train_f1_label_0": f1_train_label_0,
            "test_f1_label_1": f1_test_label_1,
            "test_f1_label_0": f1_test_label_0,
            "train_precision_label_1": precizione_train_label_1,
            "train_precision_label_0": precision_train_label_0,
            "test_precision_label_1": precision_test_label_1,
            "test_precision_label_0": precision_test_label_0,
            "train_recall_label_1": recall_train_label_1,
            "train_recall_label_0": recall_train_label_0,
            "test_recall_label_1": recall_test_label_1,
            "test_recall_label_0": recall_test_label_0
        })

# COMMANDS

@app.command()
def create_dataset(seed: int, n_features: int = 7, overwrite: bool = False, use_bert: bool = True):
    sim_train_path = SIM_TRAIN_PATH % (seed, "_w-bert" if use_bert else "")
    sim_test_path = SIM_TEST_PATH % (seed, "_w-bert" if use_bert else "")

    train,test = load_df(
        train_path=DF_TRAIN_PATH,
        test_path=DF_TEST_PATH 
    )

    if os.path.exists(sim_train_path) and not overwrite:
        console.log(f"Train data already exists at {sim_train_path}")
    else:
        console.log("Saving train data")
        save_sim_data(sim_train_path, train, n_features, oversample=True, use_bert=use_bert)

    if os.path.exists(sim_test_path) and not overwrite:
        console.log(f"Test data already exists at {sim_test_path}")
    else:
        console.log("Saving test data")
        save_sim_data(sim_test_path, test, n_features, oversample=False, use_bert=use_bert)

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

@app.command()
def nn_classify(seed: int, bert: bool = False):
    train, test = load_sim_data(
        train_path=SIM_TRAIN_PATH % (seed, "_w-bert" if bert else ""),
        test_path=SIM_TEST_PATH % (seed, "_w-bert" if bert else "")
    )

    scaler = MinMaxScaler()
    train.iloc[:, :-1] = scaler.fit_transform(train.iloc[:, :-1])
    test.iloc[:, :-1] = scaler.transform(test.iloc[:, :-1])

    wandb.init(
        project="fl-ner",
        entity="mlgroup",
        name=f"kernel-mlp-{seed}" if not bert else f"kernel-mlp-{seed}-w-bert",
        group="w-bert" if bert else "no-bert",
        tags=["flner", "centralized", "spectrum-kernel", "mlp", "w-bert" if bert else "no-bert"],
        config={
            "seed": seed,
            "train_size": len(train),
            "test_size": len(test),
            "kernel": "spectrum + w-bert",
            "n_features": train.shape[1] - 1,
            "model": "MLP"
        }
    )

    # Convert data to PyTorch tensors
    train_x = torch.tensor(train.iloc[:, :-1].values, dtype=torch.float32)
    train_y = torch.tensor(train.iloc[:, -1].values, dtype=torch.long)
    test_x = torch.tensor(test.iloc[:, :-1].values, dtype=torch.float32)
    test_y = torch.tensor(test.iloc[:, -1].values, dtype=torch.long)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = MLP(input_dim=train_x.shape[1], hidden_dim=128, output_dim=2)
    model = model.to("cuda:0")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    step = 0
    for epoch in track(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to("cuda:0"), batch_y.to("cuda:0")
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1

            if step % 10 == 0:
                model.eval()
                log_metrics(step, model, train_x, train_y, test_x, test_y, running_loss)
                model.train()
        
    # Evaluate the model
    model.eval()
    log_metrics(step, model, train_x, train_y, test_x, test_y, running_loss, print_classification_report=True)
        

if __name__ == "__main__":
    app()
