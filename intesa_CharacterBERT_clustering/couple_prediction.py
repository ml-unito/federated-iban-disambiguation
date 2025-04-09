import os
import json
import time
import torch
import sys
import wandb
from typing import Tuple, Callable
from lib.plot import *
from lib.saveOutput import *
from lib.datasetManipulation import *
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from lib.download import download_pre_trained_model
from lib.trainingUtilities import EarlyStopping, SaveBestModel

download_pre_trained_model()

import lib.CBertClassif as cbert
import lib.CBertClassifFrz as cbertfr
import lib.CBertClassifFrzSep as cbertfrsp


DEBUG = False
DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
LOG_NAME = "train_log_" + DATE_NAME + ".txt"
PLOT_NAME = "./out/couple_prediction/plot/plot_train/plot_" + DATE_NAME + ".png"
BEST_MODEL_PATH = './out/couple_prediction/output_model/model_weight_' + DATE_NAME + '.pt'
LOG_WANDB = True


writeLog = SaveOutput('./out/couple_prediction/log/', LOG_NAME, printAll=False, debug=DEBUG)



def test_model(model, X_test, y_test, criterion, batch_size: int, test):
    writeLog("\nTesting on Test Set\n")
    
    _, metrics, predictions, total_labels = test(model, X_test, y_test, batch_size, criterion)
    
    for el in metrics: writeLog("- " + el +  ":" + str(metrics[el]))
    # if not DEBUG: plot_confusion_matrix(total_labels, predictions, ['Same name (0)', 'Different name(1)'], (7,4), saveName=PLOT_NAME) 

    return metrics


def train_model(model, optimizer, scheduler, criterion, num_epochs: int, batch_size: int, X_train, y_train, X_test, y_test, train: Callable, test: Callable):
    training_loss = []
    training_accuracy = []
    test_loss = []
    test_accuracy = []
    test_f1 = []

    # early_stopping_train = EarlyStopping(patience=3, delta=0.001)
    # early_stopping_val = EarlyStopping(patience=3, delta=0.001)
    saveCriteria = SaveBestModel(BEST_MODEL_PATH, 0.015)
    writeLog("\n\nTraining step")
    init = time.time()
    
    
    for epoch in range(num_epochs):
        # --------------------------------------
        # Training step
        # --------------------------------------
        
        writeLog("\n\n------ Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + " ------\n")
        loss_tr, metrics_tr = train(model, X_train, y_train, batch_size, optimizer, criterion, scheduler)
        training_loss.append(loss_tr)
        training_accuracy.append(metrics_tr["accuracy"])

        # --------------------------------------
        # Evaluate on test set
        # --------------------------------------
        
        loss_ts, metrics_ts, _, _ = test(model, X_test, y_test, batch_size, criterion)
        test_loss.append(loss_ts)
        test_accuracy.append(metrics_ts["accuracy"])
        test_f1.append(metrics_ts['f1'])
        saveCriteria(model, (epoch + 1), loss_tr, loss_ts)
        
        writeLog(f'- Training set loss: {loss_tr}')
        writeLog(f'- Training set accuracy: {metrics_tr["accuracy"]} \n')
        writeLog(f'- test set loss: {loss_ts}')
        for el in metrics_ts: writeLog("- " + el +  ":" + str(metrics_ts[el]))
        writeLog(" ")
        
        if LOG_WANDB:
            wandb.log({
                "train_loss": loss_tr,
                "train_accuracy": metrics_tr["accuracy"],
                "train_precision": metrics_tr["precision"],
                "train_recall": metrics_tr["recall"],
                "train_f1": metrics_tr["f1"],
                "test_loss": loss_ts,
                "test_accuracy": metrics_ts["accuracy"],
                "test_precision": metrics_ts["precision"],
                "test_recall": metrics_ts["recall"],
                "test_f1": metrics_ts["f1"],
                "epoch": epoch
            })
        
        # Evaluate The early stopping
        # if early_stopping_train(loss_t) or early_stopping_val(loss_ts):
        #     writeLog("\nEarly stopping after {} epochs!".format(epoch + 1))
        #     break
    
    end = time.time()
    writeLog("\n\nTraining time: " + str((end - init) / 60) + " minutes")

    return training_loss, test_loss, test_accuracy, test_f1


# def split_dataset(X, y, train_proportion):
#     writeLog("\n- Preview of the dataset after the tokenization step:")
#     for i in range(5):writeLog(str(X[i]))
#     writeLog("")
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, random_state=42, stratify=y)
#     X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=42, stratify=y_test)

#     # Print the class distribution
#     writeLog("Original data: " + str(Counter(y)))
#     writeLog("Training data: " + str(Counter(y_train)))
#     writeLog("Test data: " + str(Counter(y_test)))
#     writeLog("Validation data: " + str(Counter(y_val)))
#     writeLog("")
#     writeLog("Training set size: " + str(len(X_train)))
#     writeLog("Validation set size: " + str(len(X_val)))
#     writeLog("Test set size: " + str(len(X_test)))
#     writeLog("")

#     return X_train, X_test, y_train, y_test, X_val, y_val


def extract_x_and_y(df: pd.DataFrame, model_name: str, tokenizer) -> Tuple[list, list]:
    if model_name == "CBertClassifFrzSep":
        X = tokenize_dataset_pair(df, tokenizer).tolist()
    else:
        X = tokenize_dataset(df, tokenizer).tolist()
    
    y = df['label'].tolist()

    return X, y


def loads_dataset(dataset_path: str): 
    writeLog("Output Log: " + str(datetime.now()))
    
    datasetName = os.path.basename(dataset_path).split(".")[0]
    dataset_directory_path = "./dataset/Train/" + datasetName
    couple_df_path = dataset_directory_path + "/" + datasetName + "_couple.csv"
    if not os.path.exists(dataset_directory_path):os.makedirs(dataset_directory_path)
    
    dataset = load_dataset(dataset_path)
    
    writeLog("\nDataset path: " + dataset_path)
    writeLog("Dataset Preview\n" + str(writeLog(dataset.head(5).to_markdown())))

    return dataset, couple_df_path


def create_couple_dataframe(train_path: str, test_path: str, balance: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # -------------------------------------------------------
    # Load the dataset and print the preview on the log file
    # -------------------------------------------------------

    train_df, cp_train_df_path = loads_dataset(train_path)
    test_df, cp_test_df_path = loads_dataset(test_path)

    # -------------------------------------------------------
    # Preprocessing dataset.
    # Eventually - ballancing the dataset on the IsShared column
    # -------------------------------------------------------
    
    train_df = prepocess_dataset(train_df)
    test_df = prepocess_dataset(test_df)

    if balance:
        train_df = balance_dataset(train_df, "IsShared")
        writeLog("\n\nBalancing the new train dataset on the IsShared column")
        writeLog("Dataset, IsShared statistics")
        writeLog(str(train_df.groupby('IsShared').size()))

        test_df = balance_dataset(test_df, "IsShared")
        writeLog("\n\nBalancing the new test dataset on the IsShared column")
        writeLog("Dataset, IsShared statistics")
        writeLog(str(test_df.groupby('IsShared').size()))
    else:
        writeLog("\nTrain dataset, IsShared statistics")
        writeLog(str(train_df.groupby('IsShared').size()))

        writeLog("\nTest dataset, IsShared statistics")
        writeLog(str(train_df.groupby('IsShared').size()))

    # -------------------------------------------------------
    # Create the dataset of pairs for the couple prediction task
    # Eventually - ballancing the dataset on the label column
    # -------------------------------------------------------
    
    cp_train_df = create_pairs(train_df)
    cp_test_df = create_pairs(test_df)

    del train_df, test_df

    writeLog("Preview of the train dataset for the couple prediction taks:\n")
    writeLog(cp_train_df.head(10).to_markdown())
    
    if balance:
        cp_train_df = balance_dataset(cp_train_df, "label")
        writeLog("\n\nBalancing the new train dataset on the labels column")
        writeLog("Dataset, label statistics")
        writeLog(str(cp_train_df.groupby('label').size()))

        cp_test_df = balance_dataset(cp_test_df, "label")
        writeLog("\n\nBalancing the new test dataset on the labels column")
        writeLog("Dataset, label statistics")
        writeLog(str(cp_test_df.groupby('label').size()))
    else:
        writeLog("\nTrain dataset, label statistics")
        writeLog(str(cp_train_df.groupby('label').size()))

        writeLog("\nTest dataset, label statistics")
        writeLog(str(cp_test_df.groupby('label').size()))

    writeLog("\n\n- Preview of the dataset before the tokenization step:")
    writeLog(cp_train_df['text'].head(5).to_markdown())
    writeLog("")
    
    cp_train_df.to_csv(cp_train_df_path, index=False)
    cp_test_df.to_csv(cp_test_df_path, index=False)

    return cp_train_df, cp_test_df


def couple_prediction(model, tokenizer, train_path: str, test_path: str, balance: bool, parameters: dict, train: Callable, test: Callable, name_wandb: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    writeLog("")
    for el in parameters: writeLog("- " +str(el) + " - " + str(parameters[el]))
    writeLog("\n\nModel:\n")
    writeLog(str(model))
    writeLog("\n")

    cp_train_df, cp_test_df = create_couple_dataframe(train_path, test_path, balance)

    model_name = model. __class__. __name__
    X_train, y_train = extract_x_and_y(df=cp_train_df, model_name=model_name, tokenizer=tokenizer)
    X_test, y_test = extract_x_and_y(df=cp_test_df, model_name=model_name, tokenizer=tokenizer)

    # -------------------------------------------------------
    # Train the model
    # -------------------------------------------------------
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(len(X_train) // parameters['batch_size'])*parameters['num_epochs'])
    criterion = torch.nn.CrossEntropyLoss()

    if LOG_WANDB:
        wandb.init(
            project="fl-ner",
            entity="mlgroup",
            tags=["flner", "test", "centralize"],
            name=name_wandb,
            config={
                "batch_size": parameters["batch_size"],
                "epochs": parameters["num_epochs"],
                "weight_decay": parameters["weight_decay"],
                "learning_rate": parameters["learning_rate"],
                "model": str(model),
                "loss": str(criterion),
                "optimizer": str(optimizer),
                "tokenizer": str(tokenizer.name_or_path),
                "train_dataset": {
                    "path": train_path,
                    "num_couple": len(cp_train_df)
                },
                "test_dataset": {
                    "path": test_path,
                    "num_couple": len(cp_test_df)
                }
            }
        )
    
    del cp_train_df, cp_test_df

    training_loss, test_loss, test_accuracy, test_f1 = train_model(model, optimizer, scheduler, criterion, parameters['num_epochs'], parameters['batch_size'], X_train, y_train, X_test, y_test, train, test)

    # --------------------------------------
    # Evaluate on test set
    # --------------------------------------

    metrics = test_model(model, X_test, y_test, criterion, parameters['batch_size'], test)

    if LOG_WANDB:
        wandb.log({
            "final_test_accuracy": metrics["accuracy"],
            "final_test_precision": metrics["precision"],
            "final_test_recall": metrics["recall"],
            "final_test_f1": metrics["f1"]
        })
        wandb.summary["test_accuracy"] = metrics["accuracy"]
        wandb.summary["test_precision"] = metrics["precision"]
        wandb.summary["test_recall"] = metrics["recall"]
        wandb.summary["test_f1"] = metrics["f1"]
        wandb.finish()
    
    # -------------------------------------------------------
    # Plot the metrics
    # -------------------------------------------------------
    
    plot_metrics(training_loss, test_loss, test_accuracy, test_f1, (9,4), saveName=PLOT_NAME)
    writeLog("\nYou can see the plot at the link: " + PLOT_NAME)
    writeLog("You can see the model .pt file at the link: " + BEST_MODEL_PATH)


def main(model_name: str, train_path: str, test_path: str, config_path: str, balance: bool, name_wandb: str):
    # Loads parameters
    config_file = open(config_path)
    parameters = json.load(config_file)

    # Loads tokenizer
    tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')

    if model_name == "CBertClassif":
        couple_prediction(model=cbert.CBertClassif(), tokenizer=tokenizer, train_path=train_path, test_path=test_path, balance=balance, parameters=parameters,
                          train=cbert.train, test=cbert.test, name_wandb=name_wandb)
    elif model_name == "CBertClassifFrz":
        couple_prediction(model=cbertfr.CBertClassifFrz(), tokenizer=tokenizer, train_path=train_path, test_path=test_path, balance=balance, parameters=parameters,
                          train=cbertfr.train, test=cbertfr.test, name_wandb=name_wandb)
    elif model_name == "CBertClassifFrzSep":
        couple_prediction(model=cbertfrsp.CBertClassifFrzSep(), tokenizer=tokenizer, train_path=train_path, test_path=test_path, balance=balance, parameters=parameters,
                          train=cbertfrsp.train, test=cbertfrsp.test, name_wandb=name_wandb)
    else:
        print("Error: unknown model.")
    

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("\nType a dataset, first!")
        print("USAGE: python3 couple_prediction.py TRAIN_PATH TEST_PATH MODEL CONFIG_PATH NAME_LOG")
        print("where, TRAIN_PATH is a .csv or .xlsx file of train dataset")
        print("where, TEST_PATH is a .csv or .xlsx file of test dataset")
        print("where, MODEL is the name of the model")
        print("where, NAME_LOG is the name of the wandb log")
        exit()

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_name = sys.argv[3]
    config_path = sys.argv[4]
    name_wandb = sys.argv[5]

    balance = True
    if len(sys.argv) == 7:
        if(sys.argv[6] == "unbalance"):
            balance = False
    
    main(model_name=model_name, train_path=train_path, test_path=test_path,
          config_path=config_path, balance=balance, name_wandb=name_wandb)