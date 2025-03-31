import os
import json
import time
import torch
import sys
import wandb
from lib.plot import *
from lib.saveOutput import *
from collections import Counter
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


# New log File
writeLog = SaveOutput('./out/couple_prediction/log/', LOG_NAME, printAll=False, debug=DEBUG)



def test_model(model, X_test, y_test, criterion, batch_size: int, test):
    writeLog("\nTesting on Test Set\n")
    
    _, metrics, predictions, total_labels = test(model, X_test, y_test, batch_size, criterion)
    
    for el in metrics: writeLog("- " + el +  ":" + str(metrics[el]))
    # if not DEBUG: plot_confusion_matrix(total_labels, predictions, ['Same name (0)', 'Different name(1)'], (7,4), saveName=PLOT_NAME) 

    return metrics


def train_model(model, optimizer, scheduler, criterion, num_epochs: int, batch_size: int, X_train, y_train, X_val, y_val, train, test):
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    validation_f1 = []

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
        loss_t, metrics_t = train(model, X_train, y_train, batch_size, optimizer, criterion, scheduler)
        training_loss.append(loss_t)
        training_accuracy.append(metrics_t["accuracy"])

        # --------------------------------------
        # Evaluate on validation set
        # --------------------------------------
        
        loss_v, metrics_v, _, _ = test(model, X_val, y_val, batch_size, criterion)
        validation_loss.append(loss_v)
        validation_accuracy.append(metrics_v["accuracy"])
        validation_f1.append(metrics_v['f1'])
        saveCriteria(model, (epoch + 1), loss_t, loss_v)
        
        writeLog(f'- Training set loss: {loss_t}')
        writeLog(f'- Training set accuracy: {metrics_t["accuracy"]} \n')
        writeLog(f'- Validation set loss: {loss_v}')
        for el in metrics_v: writeLog("- " + el +  ":" + str(metrics_v[el]))
        writeLog(" ")
        
        if LOG_WANDB:
            wandb.log({
                "train_loss": loss_t,
                "train_accuracy": metrics_t["accuracy"],
                "train_precision": metrics_t["precision"],
                "train_recall": metrics_t["recall"],
                "train_f1": metrics_t["f1"],
                "train_epoch": epoch,
                "val_loss": loss_v,
                "val_accuracy": metrics_v["accuracy"],
                "val_precision": metrics_v["precision"],
                "val_recall": metrics_v["recall"],
                "val_f1": metrics_v["f1"],
                "val_epoch": epoch
            })
        
        # Evaluate The early stopping
        # if early_stopping_train(loss_t) or early_stopping_val(loss_v):
        #     writeLog("\nEarly stopping after {} epochs!".format(epoch + 1))
        #     break
    
    end = time.time()
    writeLog("\n\nTraining time: " + str((end - init) / 60) + " minutes")

    return training_loss, validation_loss, validation_accuracy, validation_f1


def create_couple_dataframe(dataset_path: str, balance: bool) -> pd.DataFrame:
    # -------------------------------------------------------
    # Load the dataset and print the preview on the log file
    # -------------------------------------------------------

    dataset, dataset_preprocessed_path = loads_dataset(dataset_path)

    # -------------------------------------------------------
    # Preprocessing dataset.
    # Eventually - ballancing the dataset on the IsShared column
    # -------------------------------------------------------
    
    dataset = prepocess_dataset(dataset)
    if balance:
        dataset = balance_dataset(dataset,"IsShared")
        writeLog("\nBalancing the dataset on the isShared column")
        writeLog("Dataset, isShared statistics:")
        writeLog(str(dataset.groupby('IsShared').size()))
    else:
        writeLog("\nDataset, isShared statistics")
        writeLog(str(dataset.groupby('IsShared').size()))

    # -------------------------------------------------------
    # Create the dataset of pairs for the couple prediction task
    # Eventually - ballancing the dataset on the label column
    # -------------------------------------------------------
    
    dataframe = create_pairs(dataset)

    del dataset

    writeLog("Preview of the dataset for the couple prediction taks:\n")
    writeLog(dataframe.head(10).to_markdown())
    
    if balance:
        dataframe = balance_dataset(dataframe, "label")
        writeLog("\n\nBalancing the new dataset on the labels column")
        writeLog("Dataset, label statistics")
        writeLog(str(dataframe.groupby('label').size()))
    else:
        writeLog("\nDataset, label statistics")
        writeLog(str(dataframe.groupby('label').size()))

    writeLog("\n\n- Preview of the dataset before the tokenization step:")
    writeLog(dataframe['text'].head(5).to_markdown())
    writeLog("")
    
    dataframe.to_csv(dataset_preprocessed_path, index=False)

    return dataframe
    

def split_dataset(X, y, train_proportion):
    writeLog("\n- Preview of the dataset after the tokenization step:")
    for i in range(5):writeLog(str(X[i]))
    writeLog("")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=42, stratify=y_test)

    # Print the class distribution
    writeLog("Original data: " + str(Counter(y)))
    writeLog("Training data: " + str(Counter(y_train)))
    writeLog("Test data: " + str(Counter(y_test)))
    writeLog("Validation data: " + str(Counter(y_val)))
    writeLog("")
    writeLog("Training set size: " + str(len(X_train)))
    writeLog("Validation set size: " + str(len(X_val)))
    writeLog("Test set size: " + str(len(X_test)))
    writeLog("")

    return X_train, X_test, y_train, y_test, X_val, y_val


def loads_dataset(dataset_path: str): 
    writeLog("Output Log: " + str(datetime.now()))
    
    datasetName = os.path.basename(dataset_path).split(".")[0]
    dataset_directory_path = "./dataset/Train/" + datasetName
    dataset_preprocessed_path = dataset_directory_path + "/" + datasetName + "_couple.csv"
    if not os.path.exists(dataset_directory_path):os.makedirs(dataset_directory_path)
    
    dataset = load_dataset(dataset_path)
    
    writeLog("\nDataset path: " + dataset_path)
    writeLog("Dataset Preview\n" + str(writeLog(dataset.head(5).to_markdown())))

    return dataset, dataset_preprocessed_path


def couple_prediction(model, tokenizer, dataset_path: str, balance: bool, parameters: dict, train, test, name_wandb: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    writeLog("")
    for el in parameters: writeLog("- " +str(el) + " - " + str(parameters[el]))
    writeLog("\n\nModel:\n")
    writeLog(str(model))
    writeLog("\n")

    dataframe = create_couple_dataframe(dataset_path, balance)
    
    if model. __class__. __name__ == "CBertClassifFrzSep":
        X = tokenize_dataset_pair(dataframe, tokenizer).tolist()
    else:
        X = tokenize_dataset(dataframe, tokenizer).tolist()
    y = dataframe['label'].tolist()
    del dataframe

    X_train, X_test, y_train, y_test, X_val, y_val = split_dataset(X, y, parameters["train_proportion"])

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
                "train_proportion": parameters["train_proportion"],
                "dataset": dataset_path
            }
        )

    training_loss, validation_loss, validation_accuracy, validation_f1 = train_model(model, optimizer, scheduler, criterion, parameters['num_epochs'], parameters['batch_size'], X_train, y_train, X_val, y_val, train, test)

    # --------------------------------------
    # Evaluate on test set
    # --------------------------------------

    metrics = test_model(model, X_test, y_test, criterion, parameters['batch_size'], test)

    if LOG_WANDB:
        wandb.log({
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"]
        })
        wandb.summary["test_accuracy"] = metrics["accuracy"]
        wandb.summary["test_precision"] = metrics["precision"]
        wandb.summary["test_recall"] = metrics["recall"]
        wandb.summary["test_f1"] = metrics["f1"]
        wandb.finish()
    
    # -------------------------------------------------------
    # Plot the metrics
    # -------------------------------------------------------
    
    plot_metrics(training_loss, validation_loss, validation_accuracy, validation_f1, (9,4), saveName=PLOT_NAME)
    writeLog("\nYou can see the plot at the link: " + PLOT_NAME)
    writeLog("You can see the model .pt file at the link: " + BEST_MODEL_PATH)


def main(model_name: str, dataset_path: str, config_path: str, balance: bool, name_wandb: str):
    # Loads parameters
    config_file = open(config_path)
    parameters = json.load(config_file)

    # Loads tokenizer
    tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')

    if model_name == "CBertClassif":
        couple_prediction(model=cbert.CBertClassif(), tokenizer=tokenizer, dataset_path=dataset_path, balance=balance, parameters=parameters,
                          train=cbert.train, test=cbert.test, name_wandb=name_wandb)
    elif model_name == "CBertClassifFrz":
        couple_prediction(model=cbertfr.CBertClassifFrz(), tokenizer=tokenizer,dataset_path=dataset_path, balance=balance, parameters=parameters,
                          train=cbertfr.train, test=cbertfr.test, name_wandb=name_wandb)
    elif model_name == "CBertClassifFrzSep":
        couple_prediction(model=cbertfrsp.CBertClassifFrzSep(), tokenizer=tokenizer,dataset_path=dataset_path, balance=balance, parameters=parameters,
                          train=cbertfrsp.train, test=cbertfrsp.test, name_wandb=name_wandb)
    else:
        print("Error: unknown model.")
    

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("\nType a dataset, first!")
        print("USAGE: python3 couple_prediction.py DATASET_PATH MODEL CONFIG_PATH BALANCE")
        print("where, DATASET_PATH is a .csv or .xlsx file")
        print("where, MODEL is the name of the model")
        print("where, NAME_LOG is the name of the wandb log")
        print("where, BALANCE for balancing the dataset. Take one of ['unbalance' or 'balance'] default balance")
        exit()

    dataset_path = sys.argv[1]
    model_name = sys.argv[2]
    config_path = sys.argv[3]
    name_wandb = sys.argv[4]

    balance = True
    if len(sys.argv) == 6:
        if(sys.argv[5] == "unbalance"):
            balance = False
    
    main(model_name=model_name, dataset_path=dataset_path, config_path=config_path, balance=balance, name_wandb=name_wandb)