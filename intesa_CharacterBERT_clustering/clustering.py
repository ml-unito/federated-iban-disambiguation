import json
import torch
import sys
import os
import wandb
import pandas as pd
from typing import Tuple, Callable
from lib.plot import *
from lib.saveOutput import *
from collections import Counter
from itertools import combinations
from lib.datasetManipulation import *
from transformers import BertTokenizer
from lib.download import download_pre_trained_model
from lib.trainingUtilities import compute_metrics
from tqdm import tqdm

download_pre_trained_model()
import lib.CBertClassif as cbert
import lib.CBertClassifFrz as cbertfrz


DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
LOG_NAME = "clustering_test_log_" + DATE_NAME + ".txt"
JSON_NAME = "clusters_" + DATE_NAME + ".json"
DATASET_BUILD = "labelled_testSet_" + DATE_NAME + ".csv"
DEBUG_MODE = False
DEVICE = "cuda:0"
LOG_WANDB = False

# parameters
saveToFile = SaveOutput('./out/clustering/Log/', LOG_NAME, printAll=False, debug=DEBUG_MODE)
with open('./config/parameters.json', "r") as data_file:
    parameters = json.load(data_file)
batch_size = parameters['batch_size']



def eval_cluster_iban_pred(dataset: pd.DataFrame):
    number_cluster_iban_ok = 0
    shared_not_clustered_iban = 0
    for iban, group in dataset.groupby('AccountNumber'):
        predicted_holder = group['Predicted_Holder'].tolist()
        holder = group['Holder'].tolist()
        check = [predicted_holder[i] == holder[i] for i in range(len(predicted_holder))]
        number_cluster_iban_ok += 1 if all(check) else 0
        if not all(check): 
            saveToFile("IBAN: " + iban + " not correctly clustered! --> " + "Transaction OK: " + str(len([el for el in check if el == True])) + " / " + str(len(check)))
            if dataset.loc[dataset['AccountNumber'] == iban]['IsShared'].tolist()[0] == 1: 
                shared_not_clustered_iban += 1
    
    return number_cluster_iban_ok, shared_not_clustered_iban
    

def eval_transaction_holder_pred(dataset: pd.DataFrame, account_entities: dict):
    '''It returns the number of entry with correct prediction fo holder.'''

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
    
    num_entry_correct_holder = len(dataset.loc[dataset['Holder'] == dataset['Predicted_Holder']])

    return num_entry_correct_holder
    

def eval_is_shared_pred(account_entities: dict):
    for iban in account_entities:
        if len(account_entities[iban]['holders']) > 1: 
            account_entities[iban]['predicted_shared'] = 1
        elif len(account_entities[iban]['holders']) == 1: 
            account_entities[iban]['predicted_shared'] = 0

    # Correction on prediction holders
    for iban in account_entities:
        if account_entities[iban]['predicted_shared'] == 1 and account_entities[iban]['IsShared'] == 0:
            for i,holder in enumerate(account_entities[iban]['holders']):
                holder['holder_from_cluster_name'] = holder['holder_from_cluster_name'] + "_" + str(i)         
    
    # Is shared accuracy
    predictions = [account_entities[iban]['predicted_shared'] for iban in account_entities]
    real = [account_entities[iban]['IsShared'] for iban in account_entities]
    
    num_iban_correct_pred_shared = 0
    for iban in account_entities:
        if account_entities[iban]['IsShared'] == account_entities[iban]['predicted_shared']:
            num_iban_correct_pred_shared += 1
    
    return real, predictions, num_iban_correct_pred_shared


def find_partial_cliques(G: nx.Graph, min_size=3, max_size=6, tolerance=1):
    partial_cliques = []

    nodes = list(G.nodes)
    for k in range(min_size, max_size + 1):
        for subset in combinations(nodes, k):
            subgraph = G.subgraph(subset)
            num_edges = subgraph.number_of_edges()
            max_edges = k * (k - 1) // 2
            missing_edges = max_edges - num_edges

            if missing_edges <= tolerance:
                score = num_edges / max_edges
                partial_cliques.append((subset, score, missing_edges))

    partial_cliques.sort(key=lambda x: (-len(x[0]), -x[1], x[2]))  # prioritize larger cliques, higher scores, fewer missing
    return partial_cliques


def create_clusters_from_cliques(G: nx.Graph, iban: str, account_entities: dict, df: pd.DataFrame):
    # cliques = find_partial_cliques(G)

    # for clique, score, missing in cliques:
    #     saveToFile(f"Clique: {clique}, Score: {score:.2f}, Missing Edges: {missing}")

    #     clique_list = list(clique)
    #     representative_name = max(clique, key=len)
    #     r_nodes = [el for el in clique_list]
    #     account_entities[iban]['holders'].append({
    #         "cluster_name": representative_name,
    #         "names_list": r_nodes,
    #         "holder_from_cluster_name": df[(df['Name'] == representative_name) & (df['AccountNumber'] == iban)]['Holder'].tolist()[0]
    #     })
    pass


def create_clusters_from_connected_components(G: nx.Graph, iban: str, account_entities: dict, df: pd.DataFrame):
    '''Create the clusters by listing the connected components and selecting the
      representative as the longest name in each cluster'''

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


def create_graph(names1: list, names2: list, predicted: list, iban) -> nx.Graph:
    '''Create the graph of connected components per IBAN.'''

    G = nx.Graph()
    G.add_nodes_from(list(set(names1) | set(names2)))
    
    for i in range(len(predicted)): # Add edges based on predictions
        if predicted[i] == 0: 
            G.add_edge(names1[i], names2[i])

    if len(predicted) != len(names1):
        saveToFile(iban +  " " + str(len(predicted)) + " " + str(len(names1)) + " " + str(len(names2)))

    return G


def clustering(couple_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    account_entities = {}

    couple_df_groupby_iban = couple_df.groupby("iban")

    for _, group in tqdm(couple_df_groupby_iban, desc="Creates clusters"):
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

        create_clusters_from_connected_components(G, iban, account_entities, df)
        # create_clusters_from_cliques(G, iban, account_entities, df)
    
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


def load_model_with_weigths(model_name: str, path_weigths_model: str):
    model = None
    test = None
    if model_name == "CBertClassif":
        model = cbert.CBertClassif().to(DEVICE)
        test = cbert.test
    elif model_name == "CBertClassifFrz":
        model = cbertfrz.CBertClassifFrz().to(DEVICE)
        test = cbertfrz.test
    else:
        print("Error: unknown model.")

    weights = torch.load(path_weigths_model, weights_only=True)["modopt"]["model"] #errore con caricamento server fluke
    model.load_state_dict(weights)
    model.eval()

    return model, test


def accounts_disambiguation(model_name: str, weights_path: str, dataset: pd.DataFrame, name_wandb: str="clustering"):
    # load model
    model, test = load_model_with_weigths(model_name, weights_path)

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
            tags=["flner", "test", "clustering", model_name],
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
    _, metrics, predictions, total_labels = test(model, X, y, batch_size, criterion)
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
    

def main(model_name: str, weights_path: str, dataset_path: str):
    # load dataset
    dataset = load_dataset(dataset_path)
    saveToFile("Output Log " + str(datetime.now()) + "\n")
    saveToFile("Dataset path: " + dataset_path)
    saveToFile("Model path: " + model_name)   
    saveToFile("Loading dataset and model...")
    saveToFile("Dataset loaded...\n")
    
    accounts_disambiguation(model_name, weights_path, dataset[:35])

    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nType model and datasets, first!")
        print("USAGE: python clustering.py MODEL_NAME WEIGHTS_PATH DATASET_PATH")
        print("where, WEIGHTS_PATH is the path of the .pt model file")
        print("where, DATASET_PATH is a .csv or .xlsx file")
        exit()
    
    main(model_name=sys.argv[1], weights_path=sys.argv[2], dataset_path=sys.argv[3])