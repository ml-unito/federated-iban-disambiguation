import json
import torch
import sys
import os
import wandb
import yaml
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
import lib.CBertClassif as cbert

app = Typer()

with open('./config/cluster_params.yaml', "r") as file:
	params = yaml.safe_load(file)


DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-") 
DEBUG_MODE = params["debug_mode"]
DEVICE = params["device"]
DIR_OUTPUT_PATH = params["dir_output_path"]
BATCH_SIZE = params["batch_size"]

LOG_WANDB = params["wandb"]["log_wandb"]
PROJECT = params["wandb"]["project"]
ENTITY = params["wandb"]["entity"]
TAGS = params["wandb"]["tags"]


SYSTEM_SEED = 12345  # Seed per replicabilità del sistema (modello, CUDA, ecc.)
def set_system_seed(seed: int = SYSTEM_SEED):
    """Fissa tutti i seed di sistema per garantire replicabilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Per multi-GPU
    # Rende le operazioni CUDA deterministiche
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Variabile d’ambiente per hash deterministico
    os.environ["PYTHONHASHSEED"] = str(seed)


def eval_cluster_iban_pred(dataset: pd.DataFrame, excluded_ibans: list, log) -> Tuple[int, int]:
    '''It returns how many ibans have been correctly clustered, that is when
    each predicted holder matches the real holder. It also returns the number
    of shared account ibans that have not been clustered correctly. Both 
    shared and unshared account transactions are analyzed in this metric.'''

    correctly_clustered_iban = 0
    wrong_clustered_shared_iban = 0
    
    for iban, group in dataset.groupby('AccountNumber'):
        if iban not in excluded_ibans:
            predicted_holder = group['Predicted_Holder'].tolist()
            holder = group['Holder'].tolist()
            check = [predicted_holder[i] == holder[i] for i in range(len(predicted_holder))]
            correctly_clustered_iban += 1 if all(check) else 0
            if not all(check): 
                log("IBAN: " + iban + " not correctly clustered! --> " + "Transaction OK: " + str(len([el for el in check if el == True])) + " / " + str(len(check)))
                if dataset.loc[dataset['AccountNumber'] == iban]['IsShared'].tolist()[0] == 1: 
                    wrong_clustered_shared_iban += 1
    
    return correctly_clustered_iban, wrong_clustered_shared_iban
    

def eval_transaction_holder_pred(dataset: pd.DataFrame, exluded_iban: list) -> int:
    '''It returns the number of entry with correct prediction of holder, excluding
    ibans in the list. Both shared and unshared account transactions are analyzed 
    in this metric.'''

    filtered_dataset = dataset[~dataset["AccountNumber"].isin(exluded_iban)]
    return (filtered_dataset['Holder'] == filtered_dataset['Predicted_Holder']).sum()


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
        
        if holder_dict and representative_names:
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
        num_holders = len(account_entities[iban]['holders'])

        # Case where iban is shared but clusters so at least 20 and are not stored
        if num_holders == 0:
            account_entities[iban]['predicted_shared'] = 1
        else:
            if num_holders > 1: 
                account_entities[iban]['predicted_shared'] = 1
                # Correction on prediction holders
                if account_entities[iban]['IsShared'] == 0:
                    for i,holder in enumerate(account_entities[iban]['holders']):
                        holder['holder_from_cluster_name'] = holder['holder_from_cluster_name'] + "_" + str(i)
            else: 
                account_entities[iban]['predicted_shared'] = 0
            
            for index, row in dataset.loc[dataset['AccountNumber'] == iban].iterrows():
                dataset.loc[index,"IsShared_pred"] = account_entities[iban]['predicted_shared']


def create_graph(names1: list, names2: list, predicted: list, iban, log) -> nx.Graph:
    '''Create the graph of connected components per IBAN.'''

    G = nx.Graph()
    G.add_nodes_from(list(set(names1) | set(names2)))
    
    for i in range(len(predicted)): # Add edges based on predictions
        if predicted[i] == 0: G.add_edge(names1[i], names2[i])

    if len(predicted) != len(names1):
        log(iban +  " " + str(len(predicted)) + " " + str(len(names1)) + " " + str(len(names2)))

    return G


def clustering(couple_df: pd.DataFrame, df: pd.DataFrame, log) -> Tuple[dict, list]:
    account_entities = {}
    excluded_ibans = []

    couple_df_groupby_iban = couple_df.groupby("iban")
    for iban, group in couple_df_groupby_iban:
        shared = group['IsShared'].iloc[0]
        names1 = group['name1'].tolist()
        names2 = group['name2'].tolist()
        predicted = group['predicted'].tolist()
        account_entities[iban] = {
            'IsShared': int(shared),
            'predicted_shared': -1,
            'real_holders': list(set(df[df['AccountNumber'] == iban]['Holder'].tolist())),
            'holders': []
        }

        G = create_graph(names1, names2, predicted, iban, log)

        # Create the clusters by listing the connected components
        # and selecting the representative as the longest name in each cluster
        clusters = list(nx.connected_components(G))

        if len(clusters) < 20:
            for cluster in clusters:
                cluster_list = list(cluster)
                representative_name = max(cluster, key=len)
                r_nodes = [el for el in cluster_list]
                account_entities[iban]['holders'].append({
                        "cluster_name": representative_name,
                        "names_list": r_nodes,
                        "holder_from_cluster_name": df[(df['Name'] == representative_name) & (df['AccountNumber'] == iban)]['Holder'].tolist()[0]
                })
        else:
            excluded_ibans.append(iban)
    
    return account_entities, excluded_ibans


def extract_x_and_y(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')

    tokenized_texts = tokenize_dataset(df, tokenizer)
    x, y = cbert.lookup_table(tokenized_texts, df)

    return x, y


def create_pairs_for_clustering(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create pairs of names with their labels. In the returned dataframe, the
     following columns are present: "iban", "text", "name1", "name2", "label",
     "IsShared". """
    
    pairs = []
    labels = [] # The “label” column indicates whether the couple refers to the same holder (label 0) or not (label 1)
    ibans = []
    names1 = []
    names2 = []
    isShared = []
    grouped = dataset.groupby('AccountNumber')

    for iban, group in grouped:
        names = group['Name'].tolist()
        clusters = group['cluster'].tolist()
        shared = group['IsShared'].iloc[0]
        
        if(len(names)) == 1:
            pairs.append("@".join([names[0], names[0]]))
            labels.append(0)
            ibans.append(iban)
            isShared.append(shared)
            names1.append(names[0])
            names2.append(names[0])
        else:
            for (name1, cluster1), (name2, cluster2) in combinations(zip(names, clusters), 2):
                pairs.append("@".join([name1, name2]))
                labels.append(0 if cluster1 == cluster2 else 1)
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


def create_pairs_kernel_clustering(dataset: pd.DataFrame) -> pd.DataFrame:
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
def cbert_accounts_disambiguation(seed: int, weights_path: str, dataset_path: str, name_wandb: str="clustering"):
    set_system_seed()
    
    # Create output directory
    path = DIR_OUTPUT_PATH + "clustering-cbert-S" + str(seed) + "_" + DATE_NAME + "/"
    os.makedirs(path)

    log = SaveOutput(path, "log.txt", printAll=True, debug=DEBUG_MODE)
    log("Output Log " + str(datetime.now()) + "\n")
    log("Dataset path (seed "+str(seed)+"): " + dataset_path)
    log("Weights model path: "+ weights_path)

    # load dataset
    dataset = pd.read_csv(dataset_path, index_col=0)

    log("Info dataset: " 
               + str(dataset[dataset['IsShared'] == 1]['AccountNumber'].nunique()) + " shared iban, " 
               + str(dataset[dataset['IsShared'] == 0]['AccountNumber'].nunique()) + " unshared iban, "
               + str(len(dataset)) + " entries")
    
    original_dataset = dataset.copy(deep=True)
    #dataset = dataset.drop_duplicates(subset=["AccountNumber","Name","num occorrenze","IsShared","Holder","cluster"])

    # Create pairs
    pairs_df = create_pairs_for_clustering(dataset)

    # load model
    model = cbert.CBertClassif().to(DEVICE)
    weights = torch.load(weights_path, weights_only=True)
    if "model" in weights:
        model.load_state_dict(weights["model"]) #fluke
    else:
        model.load_state_dict(weights)          #flower
    model.eval()

    if LOG_WANDB:
        wandb.init(
            project=PROJECT,
            entity=ENTITY,
            tags=["flner", "clustering", "CBertClassif", str(seed), "no-complex-iban"] + TAGS,
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
                "entries": len(original_dataset),
                "pairs": len(pairs_df),
                "output_path": path
            }
        )
    

    # Tokenize pairs
    test_x, test_y = extract_x_and_y(pairs_df)
    pairs_df = pairs_df.drop('text', axis=1)
    
    log("\nTokenized text:")
    for i in range(10): log(str(test_x[i]))
    log("")
    #log("dataset proportion: " + str(Counter(test_y)))


    # Couple prediction task
    log("\n\nEvaluation of the model on test set on the couple prediction task...") 
    
    criterion = torch.nn.CrossEntropyLoss()
    _, metrics, predictions, total_labels = cbert.test(model, test_x, test_y, BATCH_SIZE, criterion)
    log(str(metrics))
    
    predictions = torch.stack(predictions).argmax(dim=1).cpu().numpy()
    cr_test = classification_report(total_labels, predictions, output_dict=True)
    cr_test_str = classification_report(total_labels, predictions, output_dict=False)
    print(cr_test_str)
    
    if LOG_WANDB:
        wandb.log({"couple_prediction": cr_test})
    
    pairs_df['predicted'] = predictions

    # Clustering
    account_entities, excluded_ibans = clustering(pairs_df, dataset, log)
    set_predicted_shared_value(dataset, account_entities)
    set_holder_predicted(dataset, account_entities)
    
    # Evaluate method on is shared prediction
    real, predictions, num_iban_correct_pred_shared = eval_is_shared_pred(account_entities)
    
    # Evaluate method on transaction holder prediction / Exact Holder prediction
    num_correct_transaction = eval_transaction_holder_pred(dataset, excluded_ibans)
    
    # Evaluate method on clustered Iban prediction
    correctly_clustered_iban, wrong_clustered_shared_iban = eval_cluster_iban_pred(dataset, excluded_ibans, log)
        
    # Print statistics
    log("\n"+str(len(excluded_ibans))+" complex ibans exluded (more than 20 holders):\n"+str(excluded_ibans))

    num_iban = len(pairs_df.groupby("iban"))
    num_filtered_iban = num_iban - len(excluded_ibans)
    num_filtered_transaction = (~dataset['AccountNumber'].isin(excluded_ibans)).sum().item()
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

    results = {
        "is_shared_task": {
            "num_iban_correct_pred": num_iban_correct_pred_shared,
            "metrics": isshared_metrics_str
        },
        "cluster_analysis": {
            "num_correctly_clustered_iban": correctly_clustered_iban,
            "num_wrong_clustered_shared_iban": wrong_clustered_shared_iban,
            "num_wrong_clustered_unshared_iban": num_filtered_iban - correctly_clustered_iban - wrong_clustered_shared_iban,
            "accuracy": correctly_clustered_iban / num_filtered_iban
        },
        "transaction_analysis": {
            "num_entry_correct_pred": num_correct_transaction,
            "accuracy": num_correct_transaction / num_filtered_transaction
        }
    }

    if LOG_WANDB:
        wandb.log(results)
        wandb.summary = results
        wandb.finish()

    log("\n\nEvaluation of the model on the IsShared classification task...")
    log("Number of iban correctly predicted: " + str(num_iban_correct_pred_shared))
    log("Number of iban: " + str(num_iban))

    for el in isshared_metrics_str: log("- " + el +  ":" + str(isshared_metrics_str[el]))
    
    log("\n\nEvaluation of the model on the correct clustered iban prediction...")
    log("Number of correctly clustered iban: " + str(correctly_clustered_iban))
    log("Number of iban: " + str(num_filtered_iban))    
    if num_filtered_iban - correctly_clustered_iban > 0:
        log("Number of wrong clustered shared iban: " + str(wrong_clustered_shared_iban))
        log("Number of wrong clustered not shared iban: " + str(results["cluster_analysis"]["num_wrong_clustered_unshared_iban"]))
    log("- Correct Clustered Iban Accuracy (correctly clustered iban / iban): " + str(results["cluster_analysis"]["accuracy"]))
    
    log("\n\nEvaluation of the model on the correct transaction prediction...")
    log("Number of transaction exactly predicted: " + str(num_correct_transaction))
    log("Number of transaction: " + str(num_filtered_transaction))    
    log("- Transaction Holder Accuracy (correct transaction / transaction): " + str(results["transaction_analysis"]["accuracy"]))


    print("Exporting results...")

    # Save couple prediction dataset
    pairs_df.to_csv(path + "labeled_couple_dataset.csv", index=False)    

    # Save original labelled dataset
    set_predicted_shared_value(original_dataset, account_entities)
    set_holder_predicted(original_dataset, account_entities)
    original_dataset.to_csv(path + "labeled_original_dataset.csv", index=False)

    # Save clusters on json file
    json.dump(account_entities, open(path + "clusters.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    return results
    

@app.command()
def kernel_accounts_disambiguation(seed: int, weights_path: str, dataset_path: str, name_wandb: str="clustering"):
    # Create output directory
    path = DIR_OUTPUT_PATH + "clustering-kernel-S" + str(seed) + "_" + DATE_NAME + "/"
    os.makedirs(path)
    
    log = SaveOutput(path, "log.txt", printAll=True, debug=DEBUG_MODE)
    
    log("Output Log " + str(datetime.now()) + "\n")
    log("Dataset path (seed "+str(seed)+"): " + dataset_path)
    log("Weights model path: "+ weights_path)

    # load dataset
    dataset = pd.read_csv(dataset_path, index_col=0)

    log("Info dataset: " 
               + str(dataset[dataset['IsShared'] == 1]['AccountNumber'].nunique()) + " shared iban, " 
               + str(dataset[dataset['IsShared'] == 0]['AccountNumber'].nunique()) + " unshared iban, "
               + str(len(dataset)) + " entries")

    original_dataset = dataset.copy(deep=True)

    dataset = dataset.drop_duplicates(subset=["AccountNumber","Name","num occorrenze","IsShared","Holder","cluster"])

    pairs_df = create_pairs_kernel_clustering(dataset)
    similarity = create_sim_data(pairs_df[['name1', 'name2', 'label']], 7)
    pairs_df = pd.concat([pairs_df, similarity.drop(columns=["label"])], axis=1)

    # load model
    weights = torch.load(weights_path, weights_only=True)
    model = MLP(input_dim=7).to(DEVICE)
    if "model" in weights:
        model.load_state_dict(weights["model"]) #fluke
    else:
        model.load_state_dict(weights)          #flower
    model.eval()

    if LOG_WANDB:
        wandb.init(
            project=PROJECT,
            entity=ENTITY,
            tags=["flner", "clustering", str(seed), "kernel-mlp", "no-complex-iban"] + TAGS,
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
                "entries": len(original_dataset),
                "output_path": path
            }
        )
    
    scaler = MinMaxScaler()
    similarity.iloc[:, :-1] = scaler.fit_transform(similarity.iloc[:, :-1])

    # Convert data to PyTorch tensors
    test_x = torch.tensor(similarity.iloc[:, :-1].values, dtype=torch.float32)
    test_y = torch.tensor(similarity.iloc[:, -1].values, dtype=torch.long)

    # Couple prediction task
    log("\n\nEvaluation of the model on test set on the couple prediction task...") 
    with torch.no_grad():
        test_preds = model(test_x.to(DEVICE)).argmax(dim=1).cpu().numpy()
        cr_test = classification_report(
            test_y.numpy(), test_preds, output_dict=True)
        cr_test_str = classification_report(
            test_y.numpy(), test_preds, output_dict=False)
        print(cr_test_str)

        if LOG_WANDB:
            wandb.log({"couple_prediction": cr_test})
        
        pairs_df['predicted'] = test_preds
    
    # Clustering
    account_entities, excluded_ibans = clustering(pairs_df, dataset, log)
    set_predicted_shared_value(dataset, account_entities)
    set_holder_predicted(dataset, account_entities)

    # Evaluate method on is shared prediction
    real, predictions, num_iban_correct_pred_shared = eval_is_shared_pred(account_entities)
    
    # Evaluate method on transaction holder prediction / Exact Holder prediction
    num_correct_transaction = eval_transaction_holder_pred(dataset, excluded_ibans)
    
    # Evaluate method on clustered Iban prediction
    correctly_clustered_iban, wrong_clustered_shared_iban = eval_cluster_iban_pred(dataset, excluded_ibans, log)
        
    # Print statistics
    log("\n"+str(len(excluded_ibans))+" complex ibans exluded (more than 20 holders):\n"+str(excluded_ibans))

    num_iban = len(pairs_df.groupby("iban"))
    num_filtered_iban = num_iban - len(excluded_ibans)
    num_filtered_transaction = (~dataset['AccountNumber'].isin(excluded_ibans)).sum().item()
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

    results = {
        "is_shared_task": {
            "num_iban_correct_pred": num_iban_correct_pred_shared,
            "metrics": isshared_metrics_str
        },
        "cluster_analysis": {
            "num_correctly_clustered_iban": correctly_clustered_iban,
            "num_wrong_clustered_shared_iban": wrong_clustered_shared_iban,
            "num_wrong_clustered_unshared_iban": num_filtered_iban - correctly_clustered_iban - wrong_clustered_shared_iban,
            "accuracy": correctly_clustered_iban / num_filtered_iban
        },
        "transaction_analysis": {
            "num_entry_correct_pred": num_correct_transaction,
            "accuracy": num_correct_transaction / num_filtered_transaction
        }
    }

    if LOG_WANDB:
        wandb.log(results)
        wandb.summary = results
        wandb.finish()

    log("\n\nEvaluation of the model on the IsShared classification task...")
    log("Number of iban correctly predicted: " + str(num_iban_correct_pred_shared))
    log("Number of iban: " + str(num_iban))

    for el in isshared_metrics_str: log("- " + el +  ":" + str(isshared_metrics_str[el]))
    
    log("\n\nEvaluation of the model on the correct clustered iban prediction...")
    log("Number of correctly clustered iban: " + str(correctly_clustered_iban))
    log("Number of iban: " + str(num_filtered_iban))    
    if num_filtered_iban - correctly_clustered_iban > 0:
        log("Number of wrong clustered shared iban: " + str(wrong_clustered_shared_iban))
        log("Number of wrong clustered not shared iban: " + str(results["cluster_analysis"]["num_wrong_clustered_unshared_iban"]))
    log("- Correct Clustered Iban Accuracy (correctly clustered iban / iban): " + str(results["cluster_analysis"]["accuracy"]))
    
    log("\n\nEvaluation of the model on the correct transaction prediction...")
    log("Number of transaction exactly predicted: " + str(num_correct_transaction))
    log("Number of transaction: " + str(num_filtered_transaction))    
    log("- Transaction Holder Accuracy (correct transaction / transaction): " + str(results["transaction_analysis"]["accuracy"]))


    print("Exporting results...")

    # Save couple prediction dataset
    pairs_df.to_csv(path + "labeled_couple_dataset.csv", index=False)    

    # Save original labelled dataset
    set_predicted_shared_value(original_dataset, account_entities)
    set_holder_predicted(original_dataset, account_entities)
    original_dataset.to_csv(path + "labeled_original_dataset.csv", index=False)

    # Save clusters on json file
    json.dump(account_entities, open(path + "clusters.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    return results


if __name__ == "__main__":
    app()
