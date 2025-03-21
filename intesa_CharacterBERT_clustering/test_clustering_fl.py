import yaml
import torch
import json
import pandas as pd
import networkx as nx
from datetime import datetime
from itertools import combinations
from lib.datasetManipulation import *
from lib.CBertClassif import CBertClassif
from lib.CBertClassif import *
from fluke.utils.log import Log


with open('./config/exp.yaml', "r") as exp_file:
	config_exp = yaml.safe_load(exp_file)
    
NUM_CLIENTS = config_exp["protocol"]["n_clients"]
DEVICE = config_exp["exp"]["device"]
PATH_TEST_DATASET = "./dataset/test_clustering.csv"
PATH_MODELS_DIR = "./out/federated_learning_models/"



def create_pairs_for_clustering(dataset: pd.DataFrame) -> pd.DataFrame:
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
            pairs.append(" @ ".join([names[0], names[0]]))
            labels.append(0)
            ibans.append(iban)
            isShared.append(shared)
            names1.append(names[0])
            names2.append(names[0])
        else:
            for (name1, holder1), (name2, holder2) in combinations(zip(names, holders), 2):
                if(isinstance(name1, float) or isinstance(name2, float)):
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


def clustering(model) -> dict:
	# load dataset
	dataset = pd.read_csv(PATH_TEST_DATASET)
	dataset = prepocess_dataset(dataset)
	
	# create pairs
	dataframe = create_pairs_for_clustering(dataset)
	
	# tokenize pairs
	X = tokenize_dataset(dataframe).tolist()
	y = dataframe['label'].tolist()
	dataframe = dataframe.drop('text', axis=1)
	
	
	# Evaluate model
	criterion = torch.nn.BCELoss()
	batch_size = 512
	_, _, predictions, _ = test(model, X, y, batch_size, criterion)
	
	# print results
	dataframe['predicted'] = predictions
	
	# Free memory
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
					print(iban +  " " + str(len(predicted)) + " " + str(len(names1)) + " " + str(len(names2)))


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
	
	# number_cluster_ok = 0
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
			# print("IBAN: " + iban + " not correctly clustered! --> " + "Transaction OK: " + str(len([el for el in check if el == True])) + " / " + str(len(check)))
			if dataset.loc[dataset['AccountNumber'] == iban]['IsShared'].tolist()[0] == 1: shared_not_clustered_iban += 1

	correct_clustered_iban = number_cluster_iban_ok / len(dataframe_groupped)
	transaction_holder_accuracy = number_transaction_ok / len(dataset)
	
	performance = {
			"dataset_path": PATH_TEST_DATASET,
			"number_of_iban": len(dataframe_groupped),
			"number_of_correct_clustered_iban": number_cluster_iban_ok,
			"perc_correct_clustered_iban": correct_clustered_iban,
			"transaction_holder_accuracy": transaction_holder_accuracy,
	}

	return performance


def load_model_with_weigths(path_weigths_model: str) -> CBertClassif:
	model = CBertClassif().to(DEVICE)
	model.load_state_dict(torch.load(path_weigths_model, weights_only=False)["model"])
	model.eval()

	return model


def main():
	general_performance = {}

	# Test clustering on clients models
	for client in range(NUM_CLIENTS):
		model = load_model_with_weigths(PATH_MODELS_DIR + "client_" + str(client) + ".pth")
		performance = clustering(model)
		general_performance["client"+str(client)] = performance
		Log.pretty_log(Log, data=performance, title="Clustering performance for client: " + str(client)) 

	# Test clustering on server model
	model = load_model_with_weigths(PATH_MODELS_DIR + "server.pth")
	performance = clustering(model)
	general_performance["server"] = performance
	
	# Prints and saves logs
	Log.pretty_log(Log, data=performance, title="Clustering performance for server: ")
	
	log_file = open("./out/federated_learning_logs/log_clustering_"+ str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + ".json", "w")
	json.dump(general_performance, log_file, indent=4)
	

    

    
    






if __name__ == "__main__":
  main()