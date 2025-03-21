import json
import torch
import sys
import pandas as pd
from lib.plot import *
from lib.saveOutput import *
from collections import Counter
from itertools import combinations
from lib.datasetManipulation import *
from transformers import BertTokenizer
from lib.download import download_pre_trained_model

download_pre_trained_model()
from lib.CBertClassif import *


# Load Custom model
model = CBertClassif()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


# test name
DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
LOG_NAME = "clustering_test_log_" + DATE_NAME + ".txt"
PLOT_NAME = "./clustering/Plot/clustering_CM_matrix_" + DATE_NAME + ".png"
JSON_NAME = "./clustering/Clusters/Clusters_" + DATE_NAME + ".json"
DATASET_BUILD = "./clustering/Output_build_dataset/labelled_testSet_" + DATE_NAME + ".csv"
DEBUG_MODE = False

# parameters
saveToFile = SaveOutput('./clustering/Log/', LOG_NAME, printAll=True, debug=DEBUG_MODE)
with open('./config/parameters.json', "r") as data_file:parameters = json.load(data_file)
batch_size = parameters['batch_size']



def create_pairs_for_clustering(dataset):
    """Create pairs of names with their labels"""
    
    pairs = []
    labels = []
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
            #print(iban)
            pairs.append(" @ ".join([names[0], names[0]]))
            labels.append(0)
            ibans.append(iban)
            isShared.append(shared)
            names1.append(names[0])
            names2.append(names[0])
        else:
            for (name1, holder1), (name2, holder2) in combinations(zip(names, holders), 2):
                if(isinstance(name1, float) or isinstance(name2, float)):
                    print("BOOOO FLOAT")
                    print(iban)
                    print(name1, name2)
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




def main():
    """ """

    if len(sys.argv) < 3:
        print("\nType model and datasets, first!")
        print("USAGE: python clustering.py MODEL_PATH DATASET_PATH  group [OPTIONAL]")
        print("where, MODEL_PATH is the path of the .pt model file")
        print("where, DATASET_PATH is a .csv or .xlsx file")
        exit()
        

    # load dataset
    datasetPath = sys.argv[2]
    dataset = load_dataset(datasetPath)
    saveToFile("Output Log " + str(datetime.now()) + "\n")
    saveToFile("Dataset path: " + datasetPath)
    saveToFile("Model path: " + sys.argv[1])   
    saveToFile("Loading dataset and model...")
    saveToFile("Dataset loaded...\n")
    
    
    # load model
    saveToFile("\n\nModel: ")
    model_params = torch.load(sys.argv[1])
    if "rounds" in model_params:
        try:out = model.load_state_dict(model_params['model'])
        except:out = model.load_state_dict(model_params['model'], map_location=torch.device('cpu'))
    else: 
        try:out = model.load_state_dict(torch.load(sys.argv[1]))
        except:out = model.load_state_dict(torch.load(sys.argv[1], map_location=torch.device('cpu')))
        
    
    # print dataset
    saveToFile(str(out))
    saveToFile("Dataset and model loaded...\n")
    saveToFile("Dataset Preview\n")
    saveToFile(dataset.head(5).to_markdown())
    
    
    # preprocess dataset
    saveToFile("Pairing dataset...")
    dataset = prepocess_dataset(dataset)
    saveToFile("\ndataset, IsShared statistics")
    saveToFile(str(dataset.groupby('IsShared').size()))
    
    
    # create pairs
    dataframe = create_pairs_for_clustering(dataset)
    saveToFile("\nDataset Preview\n")
    saveToFile(dataframe.drop("text",axis=1).head(30).to_markdown())
    saveToFile("\ndataset, Label statistics")
    saveToFile(str(dataframe.groupby('label').size()))
    saveToFile("\nPreprocessed info:")
    saveToFile(dataframe['text'][0])
    saveToFile("")
    
    
    # tokenize pairs
    tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
    X = tokenize_dataset(dataframe, tokenizer).tolist()
    y = dataframe['label'].tolist()
    dataframe = dataframe.drop('text', axis=1)
    saveToFile("\nTokenized text:")
    for i in range(10): saveToFile(str(X[i]))
    saveToFile("")
    saveToFile("dataset proportion: " + str(Counter(y)))

    
    # Evaluate model
    saveToFile("Evaluation of the model on test set on the couple prediction task...")
    criterion = nn.BCELoss()
    _, metrics, predictions, total_labels = test(model, X, y, batch_size, criterion)
    saveToFile("Couple prediction task metrics:")
    for el in metrics: saveToFile("- Couple prediction - " + el +  ":" + str(metrics[el]))
    if not DEBUG_MODE: plot_confusion_matrix(total_labels, predictions, ['Same name (0)', 'Different name(1)'], (7,4), saveName=PLOT_NAME) 
    
    # print results
    dataframe['predicted'] = predictions
    saveToFile("\nDataset Preview\n")
    saveToFile(dataframe.head(30).to_markdown())
    saveToFile("")
    
    # Free memory
    del total_labels
    del X
    del y
    
    
    
    # -------------------------------------------------------
    # Starting clustering
    # -------------------------------------------------------
    
    account_entities = {}
    dataframe_groupped = dataframe.groupby("iban")
    for _, group in dataframe_groupped:
        shared = group['IsShared'].iloc[0]
        iban = group['iban'].iloc[0]
        names1 = group['name1'].tolist()
        names2 = group['name2'].tolist()
        predicted = group['predicted'].tolist()
        account_entities[iban] = {
            'IsShared': int(shared),
            'predicted_shared': -1,
            'real_holders': list(set(dataset[dataset['AccountNumber'] == iban]['Holder'].tolist())),
            'holders': []
        }
        
        
        
        # -------------------------------------------------------
        # Create the graph of connected components per IBAN
        # -------------------------------------------------------
        
        G = nx.Graph()
        G.add_nodes_from(list(set(names1) | set(names2)))
        
        for i in range(len(predicted)): # Add edges based on predictions
            if predicted[i] == 0: G.add_edge(names1[i], names2[i])

        if len(predicted) != len(names1):
            saveToFile(iban +  " " + str(len(predicted)) + " " + str(len(names1)) + " " + str(len(names2)))


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
                    "holder_from_cluster_name": dataset[(dataset['Name'] == representative_name) & (dataset['AccountNumber'] == iban)]['Holder'].tolist()[0]
            })
        
    
    # -------------------------------------------------------
    # Evaluation of the method on: 
    # - Is shared prediction
    # - Exact Holder prediction
    # - Transaction holder prediction
    # - Clustered Iban prediction
    # -------------------------------------------------------
    
    
    #### Is shared prediction
    for a in account_entities:
        if len(account_entities[a]['holders']) > 1: account_entities[a]['predicted_shared'] = 1
        elif len(account_entities[a]['holders']) == 1: account_entities[a]['predicted_shared'] = 0
            
    
    
    #### CORRECTION ON PREDICTION HOLDERS
    for a in account_entities:
        if account_entities[a]['predicted_shared'] == 1 and account_entities[a]['IsShared'] == 0:
            for i,holder in enumerate(account_entities[a]['holders']):
                holder['holder_from_cluster_name'] = holder['holder_from_cluster_name'] + "_" + str(i)
                
    
    #### ISSHARED ACCURACY
    predictions = [account_entities[el]['predicted_shared'] for el in account_entities]
    real = [account_entities[el]['IsShared'] for el in account_entities]
    
    count = 0
    for a in account_entities:
        if account_entities[a]['IsShared'] == account_entities[a]['predicted_shared']:
            count += 1
        
    
    #### TRANSACTION HOLDER PREDICTION
    dataset['Predicted_Holder'] = ["" for el in range(len(dataset))]
    dataset['Representative_name'] = ["" for el in range(len(dataset))]
    
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
            
    number_transaction_ok = len(dataset.loc[dataset['Holder'] == dataset['Predicted_Holder']])
    
    
    
    #### CLUSTERED IBAN PREDICTION
    number_cluster_iban_ok = 0
    shared_not_clustered_iban = 0
    for iban, group in dataset.groupby('AccountNumber'):
        predicted_holder = group['Predicted_Holder'].tolist()
        holder = group['Holder'].tolist()
        check = [predicted_holder[i] == holder[i] for i in range(len(predicted_holder))]
        number_cluster_iban_ok += 1 if all(check) else 0
        if not all(check): 
            saveToFile("IBAN: " + iban + " not correctly clustered! --> " + "Transaction OK: " + str(len([el for el in check if el == True])) + " / " + str(len(check)))
            if dataset.loc[dataset['AccountNumber'] == iban]['IsShared'].tolist()[0] == 1: shared_not_clustered_iban += 1
        
    
    
    # Print all
    saveToFile("\n\nEvaluation of the model on the IsShared classification task...")
    saveToFile("Number prediction IsShared OK: " + str(count))
    saveToFile("Number of iban: " + str(len(dataframe_groupped)))    
    metrics = compute_metrics(predictions, real)
    for el in metrics: saveToFile("- " + el +  ":" + str(metrics[el]))
    saveToFile("")
    
    saveToFile("\n")
    saveToFile("Evaluation of the model on the correct clustered iban prediction...")
    saveToFile("Number of iban exactly predicted: " + str(number_cluster_iban_ok))
    saveToFile("Number of iban: " + str(len(dataframe_groupped)))    
    if len(dataframe_groupped) - number_cluster_iban_ok > 0:
        saveToFile("Number of shared iban not correctly clustered: " + str(shared_not_clustered_iban))
        saveToFile("Number of not shared iban not correctly clustered: " + str(len(dataframe_groupped) - number_cluster_iban_ok - shared_not_clustered_iban))
        
    saveToFile("- Correct Clustered Iban Accuracy:" + str(number_cluster_iban_ok / len(dataframe_groupped)))
    saveToFile("")
    
    
    
    #### Final prints
    saveToFile("\n")
    saveToFile("Evaluation of the model on the correct transaction prediction...")
    saveToFile("Number of transaction exactly predicted: " + str(number_transaction_ok))
    saveToFile("Number of transaction:" + str(len(dataset)))    
    saveToFile("- Transaction Holder Accuracy:" + str(number_transaction_ok / len(dataset)))
    saveToFile("")
    
    #### Export labelled dataset
    dataset.to_csv(DATASET_BUILD, index=False)
    
    # save clusters on json file
    saveToFile("Exporting clusters on json file...")
    if DEBUG_MODE: json.dump(account_entities, open("./differency_classifier_output/Clusters/clusters_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    else: json.dump(account_entities, open(JSON_NAME, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    
    
main()