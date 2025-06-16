import json
import torch
import sys
import os
import wandb
import pandas as pd
from typer import Typer
from typing import Tuple, Callable
from lib.plot import *
from lib.saveOutput import *
from collections import Counter
from itertools import combinations
from lib.datasetManipulation import *
from transformers import BertTokenizer
from lib.trainingUtilities import compute_metrics
from lib.mlp import MLP
from sklearn.metrics import classification_report
from lib.kernel_sim_data_utils import create_sim_data
from sklearn.preprocessing import MinMaxScaler

# download_pre_trained_model()
import lib.CBertClassifFrz as cbertfrz

app = Typer()


DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
LOG_NAME = "clustering_test_log_" + DATE_NAME + ".txt"
JSON_NAME = "clusters_" + DATE_NAME + ".json"
DATASET_BUILD = "labelled_testSet_" + DATE_NAME + ".csv"
DEBUG_MODE = False
DEVICE = "cuda:0"
LOG_WANDB = True

# parameters
saveToFile = SaveOutput('./out/clustering/Log/', LOG_NAME, printAll=True, debug=DEBUG_MODE)
with open('./config/parameters.json', "r") as data_file:
    parameters = json.load(data_file)
batch_size = parameters['batch_size']



def eval_cluster_iban_pred(dataset: pd.DataFrame) -> Tuple[int, int]:
    '''It returns how many ibans have been correctly clustered, that is when
    each predicted holder matches the real holder. It also returns the number
    of shared account ibans that have not been clustered correctly. Both 
    shared and unshared account transactions are analyzed in this metric.'''

    correctly_clustered_iban = 0
    wrong_clustered_shared_iban = 0
    
    for iban, group in dataset.groupby('AccountNumber'):
        predicted_holder = group['Predicted_Holder'].tolist()
        holder = group['Holder'].tolist()
        check = [predicted_holder[i] == holder[i] for i in range(len(predicted_holder))]
        correctly_clustered_iban += 1 if all(check) else 0
        if not all(check): 
            saveToFile("IBAN: " + iban + " not correctly clustered! --> " + "Transaction OK: " + str(len([el for el in check if el == True])) + " / " + str(len(check)))
            if dataset.loc[dataset['AccountNumber'] == iban]['IsShared'].tolist()[0] == 1: 
                wrong_clustered_shared_iban += 1
    
    return correctly_clustered_iban, wrong_clustered_shared_iban
    

def eval_transaction_holder_pred(dataset: pd.DataFrame) -> int:
    '''It returns the number of entry with correct prediction of holder. Both
    shared and unshared account transactions are analyzed in this metric.'''

    return len(dataset.loc[dataset['Holder'] == dataset['Predicted_Holder']])


def set_holder_predicted(dataset: pd.DataFrame, account_entities: dict):
    '''It sets the predicted holder and the representative name for each transaction in the dataset.'''

    dataset['Predicted_Holder'] = None #["" for el in range(len(dataset))]
    dataset['Representative_name'] = None #["" for el in range(len(dataset))]
    
    for iban in account_entities:
        holder_dict = {}
        representative_names = {}
        for holder in account_entities[iban]['holders']:
            for name in holder["names_list"]:       
                holder_dict[name] = holder['holder_from_cluster_name']
                representative_names[name] = holder['cluster_name']
         
        for index, row in dataset.loc[dataset['AccountNumber'] == iban].iterrows():
            dataset.loc[index,"Predicted_Holder"] = holder_dict[row['Name']]
            dataset.loc[index,"Representative_name"] = representative_names[row['Name']]


def eval_is_shared_pred(account_entities: dict):
    '''It calculates the number of correctly predicted ibans on is shared task.'''
    
    predictions = [account_entities[iban]['predicted_shared'] for iban in account_entities]
    real = [account_entities[iban]['IsShared'] for iban in account_entities]
    
    num_iban_correct_pred_shared = 0
    for iban in account_entities:
        if account_entities[iban]['IsShared'] == account_entities[iban]['predicted_shared']:
            num_iban_correct_pred_shared += 1
    
    return real, predictions, num_iban_correct_pred_shared
    

def set_predicted_shared_value(dataset: pd.DataFrame, account_entities: dict):
    '''It sets the account as shared (label 1) if there is more than one
      cluster, otherwise it is classified as unshared (label 0). '''
    
    dataset['IsShared_pred'] = None

    for iban in account_entities:
        if len(account_entities[iban]['holders']) > 1: 
            account_entities[iban]['predicted_shared'] = 1
            
            # Correction on prediction holders
            if account_entities[iban]['IsShared'] == 0:
                for i,holder in enumerate(account_entities[iban]['holders']):
                    holder['holder_from_cluster_name'] = holder['holder_from_cluster_name'] + "_" + str(i)
        
        elif len(account_entities[iban]['holders']) == 1: 
            account_entities[iban]['predicted_shared'] = 0
        
        for index, row in dataset.loc[dataset['AccountNumber'] == iban].iterrows():
            dataset.loc[index,"IsShared_pred"] = account_entities[iban]['predicted_shared']


def create_graph(names1: list, names2: list, predicted: list, iban) -> nx.Graph:
    '''Create the graph of connected components per IBAN.'''

    G = nx.Graph()
    G.add_nodes_from(list(set(names1) | set(names2)))
    
    for i in range(len(predicted)): # Add edges based on predictions
        if predicted[i] == 0: G.add_edge(names1[i], names2[i])

    if len(predicted) != len(names1):
        saveToFile(iban +  " " + str(len(predicted)) + " " + str(len(names1)) + " " + str(len(names2)))

    return G


def clustering(couple_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    account_entities = {}

    couple_df_groupby_iban = couple_df.groupby("iban")

    for _, group in couple_df_groupby_iban:
        shared = group['IsShared'].iloc[0]
        iban = group['iban'].iloc[0]
        names1 = group['name1'].tolist()
        names2 = group['name2'].tolist()
        predicted = group['predicted'].tolist()
        account_entities[iban] = {
            'IsShared': int(shared),
            'predicted_shared': -1,
            'real_holders': list(set(df[df['AccountNumber'] == iban]['Holder'].tolist())),
            'holders': []
        }

        G = create_graph(names1, names2, predicted, iban)

        # -------------------------------------------------------
        # Create the clusters by listing the connected components
        # and selecting the representative as the longest name in each cluster
        # -------------------------------------------------------
        clusters = list(nx.connected_components(G))

        for cluster in clusters:
            cluster_list = list(cluster)
            representative_name = max(cluster, key=len)
            r_nodes = [el for el in cluster_list]
            account_entities[iban]['holders'].append({
                    "cluster_name": representative_name,
                    "names_list": r_nodes,
                    "holder_from_cluster_name": df[(df['Name'] == representative_name) & (df['AccountNumber'] == iban)]['Holder'].tolist()[0]
            })
    
    return account_entities


def extract_x_and_y(df: pd.DataFrame) -> Tuple[list, list]:
    tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
   
    X = tokenize_dataset(df, tokenizer).tolist()
    y = df['label'].tolist()

    return X, y


def create_pairs_for_clustering(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create pairs of names with their labels"""
    
    pairs = []
    labels = [] # The “label” column indicates whether the couple refers to the same holder (label 0) or not (label 1)
    ibans = []
    names1 = []
    names2 = []
    isShared = []
    grouped = dataset.groupby('AccountNumber')

    for iban, group in grouped:
        names = group['Name'].tolist()
        holders = group['Holder'].tolist()
        shared = group['IsShared'].iloc[0]
        
        if(len(names)) == 1:
            pairs.append(" @ ".join([names[0], names[0]]))
            labels.append(0)
            ibans.append(iban)
            isShared.append(shared)
            names1.append(names[0])
            names2.append(names[0])
        else:
            for (name1, holder1), (name2, holder2) in combinations(zip(names, holders), 2):
                pairs.append(" @ ".join([name1, name2]))
                labels.append(0 if holder1 == holder2 else 1)
                ibans.append(iban)
                isShared.append(shared)
                names1.append(name1)
                names2.append(name2)     
    
    df = pd.DataFrame()
    df['iban'] = ibans
    df['text'] = pairs
    df['name1'] = names1
    df['name2'] = names2
    df['label'] = labels
    df['IsShared'] = isShared
    
    return df


def create_pairs_kernel_clustering(dataset: pd.DataFrame):
    labels = [] # The “label” column indicates whether the couple refers to the same cluster (label 0) or not (label 1)
    ibans = []
    names1 = []
    names2 = []
    isShared = []
    dataset.fillna(0, inplace=True)

    for iban, group in dataset.groupby('AccountNumber'):
        names = group['Name'].tolist()
        clusters = group['cluster'].tolist()
        shared = group['IsShared'].iloc[0]
        
        if(len(names)) == 1:
            labels.append(0)
            ibans.append(iban)
            isShared.append(shared)
            names1.append(names[0])
            names2.append(names[0])
        else:
            for (name1, cluster1), (name2, cluster2) in combinations(zip(names, clusters), 2):
                labels.append(0 if cluster1 == cluster2 else 1)
                ibans.append(iban)
                isShared.append(shared)
                names1.append(name1)
                names2.append(name2)     
    
    df = pd.DataFrame()
    df['iban'] = ibans
    df['name1'] = names1
    df['name2'] = names2
    df['label'] = labels
    df['IsShared'] = isShared
    
    return df


@app.command()
def cbert_accounts_disambiguation(weights_path: str, dataset_path: str, name_wandb: str="clustering"):
    # load dataset
    dataset = pd.read_csv(dataset_path)
    saveToFile("Output Log " + str(datetime.now()) + "\n")
    saveToFile("Dataset path: " + dataset_path)
    saveToFile("Loading dataset and model...")
    saveToFile("Dataset loaded...\n")
    
    # load model
    model = cbertfrz.CBertClassifFrz().to(DEVICE)
    weights = torch.load(weights_path, weights_only=True)["modopt"]["model"]
    model.load_state_dict(weights)
    model.eval()
    
    # Preprocess dataset
    saveToFile("Pairing dataset...")
    dataset = prepocess_dataset(dataset)
    saveToFile("\ndataset, IsShared statistics")
    saveToFile(str(dataset.groupby('IsShared').size()))
    
    # Create pairs
    couple_df = create_pairs_for_clustering(dataset)
    saveToFile("\nDataset Preview\n")
    saveToFile(couple_df.drop("text",axis=1).head(30).to_markdown())
    saveToFile("\ndataset, Label statistics")
    saveToFile(str(couple_df.groupby('label').size()))
    saveToFile("\nPreprocessed info:")
    saveToFile(couple_df['text'][0])
    saveToFile("")

    if LOG_WANDB:
        wandb.init(
            project="fl-ner",
            entity="mlgroup",
            tags=["flner", "test", "clustering", "CBertClassiFrz"],
            name=name_wandb,
            config={
                "model": model,
                "weights_path": weights_path,
                "test_dataset": {
                    "size_original": len(dataset),
                    "num_couple": len(couple_df)
                }
            }
        )
    
    # Tokenize pairs
    X, y = extract_x_and_y(couple_df)
    couple_df = couple_df.drop('text', axis=1)
    
    saveToFile("\nTokenized text:")
    for i in range(10): saveToFile(str(X[i]))
    saveToFile("")
    saveToFile("dataset proportion: " + str(Counter(y)))


    # Couple prediction task
    saveToFile("Evaluation of the model on test set on the couple prediction task...")
    criterion = torch.nn.CrossEntropyLoss()
    _, metrics, predictions, total_labels = cbertfrz.test(model, X, y, batch_size, criterion)
    saveToFile("Couple prediction task metrics:")
    for el in metrics: 
        saveToFile("- Couple prediction - " + el +  ":" + str(metrics[el]))

    del total_labels, X, y

    couple_df['outputModel'] = predictions
    couple_df['predicted'] = [torch.argmax(pred).item() for pred in predictions]

    saveToFile("\nDataset Preview\n")
    saveToFile(couple_df.head(30).to_markdown())
    saveToFile("")


    # Clustering
    account_entities = clustering(couple_df, dataset)

    
    # Evaluate method on is shared prediction
    real, predictions, num_iban_correct_pred_shared = eval_is_shared_pred(account_entities)
    
    # Evaluate method on transaction holder prediction / Exact Holder prediction
    number_transaction_ok = eval_transaction_holder_pred(dataset, account_entities)
    
    # Evaluate method on clustered Iban prediction
    number_cluster_iban_ok, shared_not_clustered_iban = eval_cluster_iban_pred(dataset)

    
        
    # Print statistics
    couple_df_groupby_iban = couple_df.groupby("iban")

    saveToFile("\n\nEvaluation of the model on the IsShared classification task...")
    saveToFile("Number prediction IsShared OK: " + str(num_iban_correct_pred_shared))
    saveToFile("Number of iban: " + str(len(couple_df_groupby_iban)))    
    metrics = compute_metrics(predictions, real)
    for el in metrics: saveToFile("- " + el +  ":" + str(metrics[el]))
    saveToFile("")

    saveToFile("\n")
    saveToFile("Evaluation of the model on the correct clustered iban prediction...")
    saveToFile("Number of iban exactly predicted: " + str(number_cluster_iban_ok))
    saveToFile("Number of iban: " + str(len(couple_df_groupby_iban)))    
    if len(couple_df_groupby_iban) - number_cluster_iban_ok > 0:
        saveToFile("Number of shared iban not correctly clustered: " + str(shared_not_clustered_iban))
        saveToFile("Number of not shared iban not correctly clustered: " + str(len(couple_df_groupby_iban) - number_cluster_iban_ok - shared_not_clustered_iban))
        
    saveToFile("- Correct Clustered Iban Accuracy:" + str(number_cluster_iban_ok / len(couple_df_groupby_iban)))
    saveToFile("")
    
    saveToFile("\n")
    saveToFile("Evaluation of the model on the correct transaction prediction...")
    saveToFile("Number of transaction exactly predicted: " + str(number_transaction_ok))
    saveToFile("Number of transaction:" + str(len(dataset)))    
    saveToFile("- Transaction Holder Accuracy:" + str(number_transaction_ok / len(dataset)))
    saveToFile("")

    results = {
        "is_shared_task": {
            "num_iban_correct_pred": num_iban_correct_pred_shared,
            "num_iban": len(couple_df_groupby_iban),
            "metrics": metrics
        },
        "cluster_analysis": {
            "num_iban_correct_pred": number_cluster_iban_ok,
            "num_iban": len(couple_df_groupby_iban),
            "num_shared_iban_not_correct_clustered": shared_not_clustered_iban,
            "num_notshared_iban_not_correct_clustered": len(couple_df_groupby_iban) - number_cluster_iban_ok - shared_not_clustered_iban,
            "accuracy": number_cluster_iban_ok / len(couple_df_groupby_iban)
        },
        "transaction_analysis": {
            "num_entry_correct_pred": number_transaction_ok,
            "num_entry": len(dataset),
            "accuracy": number_transaction_ok / len(dataset)
        }
    }

    if LOG_WANDB:
        wandb.log(results)
        wandb.summary = results
        wandb.finish()
    
    # Export labelled dataset
    dataset_path = "./out/clustering/dataset_build/"
    if not os.path.exists(dataset_path):
      os.makedirs(dataset_path)
    dataset.to_csv(dataset_path+DATASET_BUILD, index=False)
    
    # Save clusters on json file
    saveToFile("Exporting clusters on json file...")
    path_clusters = "./out/clustering/clusters/"
    if not os.path.exists(path_clusters):
        os.makedirs(path_clusters)
    json.dump(account_entities, open(path_clusters + JSON_NAME, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    return results
    

@app.command()
def kernel_accounts_disambiguation(seed: int, weights_path: str, dataset_path: str, name_wandb: str="clustering"):
    # load dataset
    dataset = pd.read_csv(dataset_path, index_col=0)

    saveToFile("Output Log " + str(datetime.now()) + "\n")
    saveToFile("Dataset path: " + dataset_path)
    saveToFile("Loading dataset and model...")
    saveToFile("Dataset loaded...\n")
    saveToFile("Info dataset: " 
               + str(dataset[dataset['IsShared'] == 1]['AccountNumber'].nunique()) + " shared iban, " 
               + str(dataset[dataset['IsShared'] == 0]['AccountNumber'].nunique()) + " unshared iban, "
               + str(len(dataset)) + " entries")

    original_dataset = dataset.copy(deep=True)

    dataset = dataset.drop_duplicates(subset=["AccountNumber","Name","num occorrenze","IsShared","Holder","cluster"])

    pairs_df = create_pairs_kernel_clustering(dataset)
    similarity = create_sim_data(pairs_df[['name1', 'name2', 'label']], 7)
    pairs_df = pd.concat([pairs_df, similarity], axis=1)

    # load model
    weights = torch.load(weights_path, weights_only=True)["model"]
    model = MLP(input_dim=7).to(DEVICE)
    model.load_state_dict(weights)
    model.eval()

    if LOG_WANDB:
        wandb.init(
            project="fl-ner",
            entity="mlgroup",
            tags=["flner", "test", "clustering", str(seed), "kernel-mlp"],
            name=name_wandb,
            config={
                "model": model,
                "weights_path": weights_path,
                "seed": seed,
                "iban": {
                    "total": str(len(original_dataset.groupby("AccountNumber"))),
                    "shared": str(dataset[dataset['IsShared'] == 1]['AccountNumber'].nunique()),
                    "unshared": str(dataset[dataset['IsShared'] == 0]['AccountNumber'].nunique())
                },
                "entries": len(original_dataset)
            }
        )
    
    scaler = MinMaxScaler()
    similarity.iloc[:, :-1] = scaler.fit_transform(similarity.iloc[:, :-1])

    # Convert data to PyTorch tensors
    test_x = torch.tensor(similarity.iloc[:, :-1].values, dtype=torch.float32)
    test_y = torch.tensor(similarity.iloc[:, -1].values, dtype=torch.long)

    # Couple prediction task
    saveToFile("Evaluation of the model on test set on the couple prediction task...") 
    with torch.no_grad():
        test_preds = model(test_x.to(DEVICE)).argmax(dim=1).cpu().numpy()
        cr_test = classification_report(
            test_y.numpy(), test_preds, output_dict=True)
        cr_test_str = classification_report(
            test_y.numpy(), test_preds, output_dict=False)
        
        print(cr_test_str)
        
        test_accuracy = cr_test["accuracy"]
        test_f1 = cr_test["macro avg"]["f1-score"]
        f1_test_label_1 = cr_test["1"]["f1-score"]
        f1_test_label_0 = cr_test["0"]["f1-score"]
        precision_test_label_1 = cr_test["1"]["precision"]
        precision_test_label_0 = cr_test["0"]["precision"]
        recall_test_label_1 = cr_test["1"]["recall"]
        recall_test_label_0 = cr_test["0"]["recall"]

        print("test_accuracy: "+str(test_accuracy)+"\n"+
              "test_f1: "+str(test_f1)+"\n"+
              "f1_test_label_1: "+str(f1_test_label_1)+"\n"+
              "f1_test_label_0: "+str(f1_test_label_0)+"\n"+
              "precision_test_label_1: "+str(precision_test_label_1)+"\n"+
              "precision_test_label_0: "+str(precision_test_label_0)+"\n"+
              "recall_test_label_1: "+str(recall_test_label_1)+"\n"+
              "recall_test_label_0: "+str(recall_test_label_0))

        if LOG_WANDB:
            wandb.log({
                "couple_prediction_accuracy": test_accuracy,
                "couple_prediction_f1": test_f1,
                "couple_prediction_f1_label_1": f1_test_label_1,
                "couple_prediction_f1_label_0": f1_test_label_0,
                "couple_prediction_precision_label_1": precision_test_label_1,
                "couple_prediction_precision_label_0": precision_test_label_0,
                "couple_prediction_recall_label_1": recall_test_label_1,
                "couple_prediction_recall_label_0": recall_test_label_0
            })
        
        pairs_df['predicted'] = test_preds
    
    # Clustering
    account_entities = clustering(pairs_df, dataset)
    set_predicted_shared_value(dataset, account_entities)
    set_holder_predicted(dataset, account_entities)

    # Evaluate method on is shared prediction
    real, predictions, num_iban_correct_pred_shared = eval_is_shared_pred(account_entities)
    
    # Evaluate method on transaction holder prediction / Exact Holder prediction
    num_correct_transaction = eval_transaction_holder_pred(dataset)
    
    # Evaluate method on clustered Iban prediction
    correctly_clustered_iban, wrong_clustered_shared_iban = eval_cluster_iban_pred(dataset)
        
    # Print statistics
    couple_df_groupby_iban = pairs_df.groupby("iban")

    saveToFile("\n\nEvaluation of the model on the IsShared classification task...")
    saveToFile("Number of iban correctly predicted: " + str(num_iban_correct_pred_shared))
    saveToFile("Number of iban: " + str(len(couple_df_groupby_iban)))

    isshared_metrics = classification_report(real, predictions, output_dict=True)
    isshared_metrics_str = {
        "accuracy": round(isshared_metrics["accuracy"], 4),
        "f1" : round(isshared_metrics["macro avg"]["f1-score"],4),
        "f1_l1" : round(isshared_metrics["1"]["f1-score"],4),
        "f1_l0" : round(isshared_metrics["0"]["f1-score"],4),
        "precision_l1" : round(isshared_metrics["1"]["precision"],4),
        "precision_l0" : round(isshared_metrics["0"]["precision"],4),
        "recall_l1" : round(isshared_metrics["1"]["recall"],4),
        "recall_l0" : round(isshared_metrics["0"]["recall"],4)
    }

    for el in isshared_metrics_str: saveToFile("- " + el +  ":" + str(isshared_metrics_str[el]))
    saveToFile("")

    saveToFile("\n")
    saveToFile("Evaluation of the model on the correct clustered iban prediction...")
    saveToFile("Number of correctly clustered iban: " + str(correctly_clustered_iban))
    saveToFile("Number of iban: " + str(len(couple_df_groupby_iban)))    
    if len(couple_df_groupby_iban) - correctly_clustered_iban > 0:
        saveToFile("Number of wrong clustered shared iban: " + str(wrong_clustered_shared_iban))
        saveToFile("Number of wrong clustered not shared iban: " + str(len(couple_df_groupby_iban) - correctly_clustered_iban - wrong_clustered_shared_iban))
        
    saveToFile("- Correct Clustered Iban Accuracy (correctly clustered iban / iban): " + str(correctly_clustered_iban / len(couple_df_groupby_iban)))
    saveToFile("")
    
    saveToFile("\n")
    saveToFile("Evaluation of the model on the correct transaction prediction...")
    saveToFile("Number of transaction exactly predicted: " + str(num_correct_transaction))
    saveToFile("Number of transaction: " + str(len(dataset)))    
    saveToFile("- Transaction Holder Accuracy (correct transaction / transaction): " + str(num_correct_transaction / len(dataset)))
    saveToFile("")

    results = {
        "is_shared_task": {
            "num_iban_correct_pred": num_iban_correct_pred_shared,
            "metrics": isshared_metrics_str
        },
        "cluster_analysis": {
            "num_correctly_clustered_iban": correctly_clustered_iban,
            "num_wrong_clustered_shared_iban": wrong_clustered_shared_iban,
            "num_wrong_clustered_unshared_iban": len(couple_df_groupby_iban) - correctly_clustered_iban - wrong_clustered_shared_iban,
            "accuracy": correctly_clustered_iban / len(couple_df_groupby_iban)
        },
        "transaction_analysis": {
            "num_entry_correct_pred": num_correct_transaction,
            "accuracy": num_correct_transaction / len(dataset)
        }
    }

    if LOG_WANDB:
        wandb.log(results)
        wandb.summary = results
        wandb.finish()
    
    # Save couple prediction dataset
    couple_df_path = "./out/clustering/couple_df/"
    if not os.path.exists(couple_df_path):
      os.makedirs(couple_df_path)
    pairs_df.to_csv(couple_df_path + "couple_df_S"+str(seed)+DATE_NAME+".csv")

    # Export original labelled dataset
    dataset_path = "./out/clustering/dataset_build/"
    if not os.path.exists(dataset_path):
      os.makedirs(dataset_path)
    set_predicted_shared_value(original_dataset, account_entities)
    set_holder_predicted(original_dataset, account_entities)
    original_dataset.to_csv(dataset_path+DATASET_BUILD, index=False)
    
    # Save clusters on json file
    saveToFile("Exporting clusters on json file...")
    path_clusters = "./out/clustering/clusters/"
    if not os.path.exists(path_clusters):
        os.makedirs(path_clusters)
    json.dump(account_entities, open(path_clusters + JSON_NAME, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    return results


if __name__ == "__main__":
    app()
