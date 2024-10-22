import json
import torch
import pandas as pd
from tqdm import tqdm
from gnn_model import GNN3
from os.path import exists
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import combinations
from saveOutput import SaveOutput
from torch_geometric.data import Data
from torchmetrics import Precision, Recall, F1Score
from sklearn.model_selection import train_test_split
import embedding_generator.main.embeddings_generator as embeddings_generator
torch.set_printoptions(precision=4, threshold=2000, edgeitems=3, linewidth=200, profile='full')



# Load parameters
with open('./config/parameters.json', "r") as data_file: parameters = json.load(data_file)
DATASET_PATH = parameters["dataset_path"]
NODE_PLACEHOLDER = parameters["node_placheholder"]
TRAIN_SIZE = parameters["train_size"]					# Percentage of the dataset to be used for training
VALID_SIZE = parameters["valid_size"]					# Percentage of the test set for validation
TRAIN_PATH = DATASET_PATH.replace(".csv", "_train.csv")
VALID_PATH = DATASET_PATH.replace(".csv", "_valid.csv")
TEST_PATH = DATASET_PATH.replace(".csv", "_test.csv")
DATASET_NAME = DATASET_PATH.split("/")[-1].replace(".csv", "")
MODEL_PATH = "./saved_model/" + "model_" + DATASET_NAME + ".pt"
LOAD_MODEL_PATH = parameters["load_model_path"]
SAVE_MODEL = parameters["save_model"]
TRAIN = parameters["only_train"]
MAX_DIM_GRAPH = parameters["max_dim_graph"]
NUM_EPOCHS = parameters["num_epochs"]
FORCE_DATASET_SPLIT = parameters["force_dataset_split"]
MAX_NUM_NODES = parameters["max_num_nodes"]			   # Max number of nodes in a BATCH_SIZE	

# Load GPU
model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define a log file
DATE_NAME = str(datetime.now()).split(".")[0].replace(" ", "_") 
LOG_NAME = "Train_log_" + DATASET_NAME + DATE_NAME + ".txt"
PLOT_NAME = "Train_plot_" + DATASET_NAME + DATE_NAME + ".png"
PLOT_VALID_NAME = "Valid_plot_" + DATASET_NAME + DATE_NAME + ".png"
DEBUG = False



def plot_train_metrics(train_loss, train_accuracy, val_loss, val_accuracy, val_precision, val_recall, val_f1, figsize=(15, 10), saveName=None):
	""" plot train and validation metrics """

	epochs = range(1, len(train_loss) + 1)
	plt.figure(figsize=figsize)

	# Plot train loss
	plt.subplot(2, 2, 1)
	plt.plot(epochs, train_loss, 'b', label="Train loss")
	plt.title("Train Loss")
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.xticks(epochs)
	plt.legend()

	# Plot validation loss
	plt.subplot(2, 2, 2)
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title("Validation Loss")
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.xticks(epochs)
	plt.legend()

	# Plot accuracy
	plt.subplot(2, 2, 3)
	plt.plot(epochs, train_accuracy, 'g', label="Train accuracy")
	plt.plot(epochs, val_accuracy, 'y', label='Validation accuracy')
	plt.title("Accuracy")
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.xticks(epochs)
	plt.legend()

	# Plot precision, recall, and f1 score
	plt.subplot(2, 2, 4)
	plt.plot(epochs, val_precision, 'b', label="Precision")
	plt.plot(epochs, val_recall, 'g', label="Recall")
	plt.plot(epochs, val_f1, 'r', label="F1 Score")
	plt.title('Validation Metrics')
	plt.xlabel('Epochs')
	plt.ylabel('Metrics')
	plt.xticks(epochs)
	plt.legend()

	plt.tight_layout(w_pad=4)
	if saveName: plt.savefig(saveName)
	else: plt.savefig("./train_plot/" + PLOT_NAME)
	print("Plot saved")
	plt.show()





def generate_ground_truth(dataset_group, node_mapping):
	""" Generate the ground truth tensor for each node (dataset_group) """

	is_shared = dataset_group["IsShared"].tolist()[0]
	num_valued_nodes = len(node_mapping.keys())

	if not is_shared:
		ground_truth = torch.ones(num_valued_nodes, num_valued_nodes)
	else:
		ground_truth = torch.zeros(num_valued_nodes, num_valued_nodes)
		for node1 in node_mapping.keys():
			for node2 in node_mapping.keys():
				if node1 == node2 or dataset_group.loc[node1]["Holder"] == dataset_group.loc[node2]["Holder"]:
					ground_truth[node_mapping[node1], node_mapping[node2]] = 1

	# Adds values for padding nodes if needed
	if num_valued_nodes < MAX_DIM_GRAPH:
		num_empty_nodes = MAX_DIM_GRAPH - num_valued_nodes

		top_right_tensors_padding_nodes = torch.zeros((num_valued_nodes, num_empty_nodes))
		bottom_tensors_padding_nodes = torch.cat(
			(torch.zeros(num_empty_nodes, num_valued_nodes), torch.ones(num_empty_nodes, num_empty_nodes)),
			dim=1)

		ground_truth = torch.cat(
			(torch.cat((ground_truth, top_right_tensors_padding_nodes), dim=1), bottom_tensors_padding_nodes),
			dim=0)

	return ground_truth




def generate_edges(node_mapping):
	"""It generates the edges between all possible pairs of entries and between
	all possible pairs of padding nodes."""

	index_nodes = list(node_mapping.values())
	edges = [[couple[0], couple[1]] for couple in list(combinations(index_nodes, 2))]
	index_padding_nodes = list(range(len(node_mapping.values()), MAX_DIM_GRAPH))
	edges_padding_nodes = [[couple[0], couple[1]] for couple in list(combinations(index_padding_nodes, 2))]
	complete_edges = edges + edges_padding_nodes
	edge_index = torch.tensor(complete_edges, dtype=torch.long)

	return edge_index




def generate_nodes(group, field):
	""" Generate the nodes for each subgraph. the infos stored in every nodes 
are the padded tensors (to be passed to the encoding layer or the BERT) of the field.
 		- group: dataset group
 		- field: field to be processed (eg. Name)"""

	# Create mapping for nodes in DATA object
	mapping = {index: i for i, index in enumerate(group.index.unique())}
	shape = 0

	# Generate the padded tensors for each name in the group Add the placeholder for the padding nodes
	words = group[field].tolist()
	shape = len(words)
	if len(words) < MAX_DIM_GRAPH:
		words += [NODE_PLACEHOLDER for _ in range(MAX_DIM_GRAPH - len(words))]
	
	x = embeddings_generator.create_characterBERT_padded_tensors(words)

	return x, mapping, shape




def create_graphs(train_set):
	""" create the graphs from the dataset """

	data_list = []
	correct_edges = []
	node_shapes = []
	mapping_list = []
 
	for _, group in tqdm(train_set.groupby(["AccountNumber"], sort=False), desc="Creating graph"):
		if len(group) > MAX_DIM_GRAPH:
			print("The number of nodes (" + str(len(group)) + ") is greater than the limit (" + str(MAX_DIM_GRAPH) + "). The entries number was reduced.")
			group = group[:MAX_DIM_GRAPH]


		# Generate nodes, edges and ground truth. Each node is stored in a Data object. The Data oject are stored in a list
		node, node_mapping, shapes = generate_nodes(group, field="Name")
		node_shapes.append(shapes)
		edge_index = generate_edges(node_mapping)
		data = Data(x=node, edge_index=edge_index.t().contiguous())
		data.validate(raise_on_error=True)
		data_list.append(data)
		ground_truth = generate_ground_truth(group, node_mapping)
		correct_edges.append(ground_truth)
		mapping_list.append(node_mapping)


	return data_list, correct_edges, node_shapes, mapping_list





def init_weights(m):
    """ Initialize the weights of the model """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        print("WEIGHT INITIALIZED")






def train(train_set, valid_set):
	""" Trains the model. """

	# Load data
	data_list, ground_truth_list, node_shapes, mapping_list = create_graphs(train_set)
	data_list_valid, ground_truth_list_valid, node_shapes_valid, mapping_list_valid = create_graphs(valid_set)


	min_num_nodes = 0
	batch_size = 0
	for i, data in enumerate(data_list):
		if min_num_nodes + data.x.shape[0] <= MAX_NUM_NODES:
			min_num_nodes += data.x.shape[0]
			batch_size += 1
	
	# Load model
	model = GNN3(input_dim=768)
	model = model.to(model_device)
 
	# Init weights
	#model.apply(init_weights)
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.041, weight_decay=0.01)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
	criterion = torch.nn.BCELoss()
	
	# print some infos
	saveToFile = SaveOutput('./log/', LOG_NAME, debug=DEBUG)
	saveToFile("")
	saveToFile("Optimizer: " + str(optimizer))
	saveToFile("Scheduler: " + str(scheduler))
	saveToFile("Criterion: " + str(criterion))
	saveToFile("Minimum batch size: " + str(batch_size))
	saveToFile("Total number of graphs: " + str(len(data_list)))
	saveToFile("Total number of nodes: " + str(data_list[0].x.shape[0] * len(data_list)))
	saveToFile("", printAll=False)
	saveToFile("Training Set size: " + str(len(data_list)))
	saveToFile("Validation Set size: " + str(len(data_list_valid)))
	

	total_train_loss = []
	total_valid_loss = []
	total_train_accuracy = []
	total_valid_accuracy = []
	total_precision = []
	total_recall = []
	total_f1 = []


	# Training Loop
	for epoch in range(NUM_EPOCHS):
		
		# run a training step
		saveToFile("\n====== init epoch " + str(epoch + 1) + " ======")
		train_accuracy, train_loss = training_step(model, optimizer, criterion, scheduler, data_list,ground_truth_list, node_shapes, mapping_list, batch_size)
		valid_accuracy, valid_loss, precision, recall, f1 = validation_step(model, optimizer, criterion, scheduler, data_list_valid, ground_truth_list_valid, node_shapes_valid, mapping_list_valid, batch_size)
		
  		# print total accuracy and total loss
		train_accuracy = train_accuracy / len(data_list)
		valid_accuracy = valid_accuracy / len(data_list_valid)
		train_loss = train_loss / len(data_list)
		valid_loss = valid_loss / len(data_list_valid)
		total_precision.append(precision)
		total_recall.append(recall)
		total_f1.append(f1)
  
		saveToFile("Training performance:")
		saveToFile("Loss: " + str(train_loss))
		saveToFile("Accuracy: " + str(train_accuracy))
		saveToFile("\nValidation performance:")
		saveToFile("Loss: " + str(valid_loss))
		saveToFile("Accuracy: " + str(valid_accuracy))
		saveToFile("Precision: " + str(precision))
		saveToFile("Recall: " + str(recall))
		saveToFile("F1: " + str(f1))
		saveToFile("")
		
		# Save the loss and accuracy for each epoch	
		total_train_loss.append(train_loss)
		total_valid_loss.append(valid_loss)
		total_train_accuracy.append(train_accuracy)
		total_valid_accuracy.append(valid_accuracy)

		# Free the cache after each epoch
		torch.cuda.empty_cache()

	# plot the train and validation metrics
	plot_train_metrics(total_train_loss, total_train_accuracy, total_valid_loss, total_valid_accuracy, total_precision, total_recall, total_f1, saveName="./train_plot/" + PLOT_NAME)
	return model






def training_step(model, optimizer, criterion, scheduler, data_list,ground_truth_list, node_shapes, mapping_list, batch_size):
	""" Training step for each epoch """

	model.train()
	total_loss = 0
	total_accuracy = 0
	saveToFile = SaveOutput('./log/', LOG_NAME, debug=DEBUG)

	for i in tqdm(range(0, len(data_list), batch_size), desc="Batch"):					# For each batch
			
		# reset the gradient for each batch
		optimizer.zero_grad(set_to_none=True)
		singular_loss = 0
		singular_accuracy = 0
		
		# get the batch	
		batch = data_list[i:i+batch_size]
		shapes = node_shapes[i:i+batch_size]
		ground_truth_batch = ground_truth_list[i:i+batch_size]
		mapping_list_batch = mapping_list[i:i+batch_size]
		
		
		for index, data in enumerate(batch):						# For each graph

			# generate the BERT embeddings
			old_data = data.x.clone()
			data.x = data.x.to(model_device)
			data.x = embeddings_generator.create_characterBERT_embeddings(data.x[:shapes[index]])

			# Pad the nodes to the maximum dimension with zeros
			if shapes[index] < MAX_DIM_GRAPH:
				null_nodes = torch.zeros(size=((MAX_DIM_GRAPH - shapes[index]), data.x.shape[1]))
				null_nodes = null_nodes.to(model_device)
				data.x = torch.cat((data.x[:shapes[index]], null_nodes), dim=0)
   

			# Forward pass
			data.to(model_device)
			pred = model.forward(data)
			ground_truth = ground_truth_batch[index].to(model_device)

			# Apply binary cross-entropy loss
			loss = criterion(pred, ground_truth)
			singular_loss = torch.add(singular_loss, loss)

			# Compute accuracy
			result = torch.eq(torch.round(pred), ground_truth)
			singular_accuracy = torch.add(singular_accuracy, (torch.sum(result) / pred.numel()))
		
   
			if i >= len(data_list) - len(batch) and index == len(batch) - 1:
				saveToFile("DECISION FOR THE LAST GRAPH IN THE BATCH IN THE TRAINING STEP -----------------------------------", printAll=False)
				saveToFile("GRAPH: " + str(index), printAll=False)
				saveToFile("MAPPING_LIST: " + str(mapping_list_batch[index]), printAll=False)
				saveToFile("", printAll=False)
				saveToFile("PREDICTION:", printAll=False)
				saveToFile(pred, printAll=False)
				saveToFile("\nDECISION:", printAll=False)
				saveToFile(torch.round(pred), printAll=False)
				saveToFile("\nGROUND TRUTH:", printAll=False)
				saveToFile(ground_truth, printAll=False)
				saveToFile("------------------------------------------------------------------------", printAll=False)
				saveToFile("\n", printAll=False)
			
   
   			# Free the cache
			data.x = data.x.cpu()
			data.x = old_data
			pred = pred.cpu()
			ground_truth = ground_truth.cpu()
			del pred, ground_truth, result, loss
				
		
		batch_loss = torch.div(singular_loss, len(batch))
		total_loss += batch_loss.item()
		total_accuracy += singular_accuracy.item()
			
  		# Backward pass for each BATCH
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Gradient clipping for preventing exploding gradients
		batch_loss.backward(retain_graph=True)
		optimizer.step()
		scheduler.step()
		

	return total_accuracy, total_loss



def validation_step(model, optimizer, criterion, scheduler, data_list, ground_truth_list, node_shapes, mapping_list, batch_size):
    """ Run a validation step for each epoch """

    saveToFile = SaveOutput('./log/', LOG_NAME, debug=DEBUG)
    model.eval()
    total_loss = 0
    total_accuracy = 0
    precision = Precision(task='binary').to(model_device)
    recall = Recall(task='binary').to(model_device)
    f1 = F1Score(task='binary').to(model_device)
    
    for i in tqdm(range(0, len(data_list), batch_size), desc="Batch"):  # For each batch

        # reset the gradient for each batch
        singular_loss = 0
        singular_accuracy = 0
        batch_pred = []
        batch_ground_truth = []

        # get the batch
        batch = data_list[i:i+batch_size]
        shapes = node_shapes[i:i+batch_size]
        ground_truth_batch = ground_truth_list[i:i+batch_size]
        mapping_list_batch = mapping_list[i:i+batch_size]

        for index, data in enumerate(batch):  # For each graph

            # generate the BERT embeddings

            old_data = data.x.clone()
            data.x = data.x.to(model_device)
            data.x = embeddings_generator.create_characterBERT_embeddings(data.x[:shapes[index]])

            # Pad the nodes to the maximum dimension with zeros
            if shapes[index] < MAX_DIM_GRAPH:
                null_nodes = torch.zeros(size=((MAX_DIM_GRAPH - shapes[index]), data.x.shape[1]))
                null_nodes = null_nodes.to(model_device)
                data.x = torch.cat((data.x[:shapes[index]], null_nodes), dim=0)

            # Forward pass
            data.to(model_device)
            pred = model.forward(data)
            ground_truth = ground_truth_batch[index].to(model_device)

            # Apply binary cross-entropy loss
            loss = criterion(pred, ground_truth)
            singular_loss = torch.add(singular_loss, loss)

            # Compute accuracy
            result = torch.eq(torch.round(pred), ground_truth)
            singular_accuracy = torch.add(singular_accuracy, (torch.sum(result) / pred.numel()))
            batch_pred.append(torch.round(pred))
            batch_ground_truth.append(ground_truth)

            if i >= len(data_list) - len(batch) and index == len(batch) - 1:
                saveToFile("DECISION FOR THE LAST GRAPH IN THE BATCH IN THE VALIDATION STEP -----------------------------------", printAll=False)
                saveToFile("GRAPH: " + str(index), printAll=False)
                saveToFile("MAPPING_LIST: " + str(mapping_list_batch[index]), printAll=False)
                saveToFile("", printAll=False)
                saveToFile("PREDICTION:", printAll=False)
                saveToFile(pred, printAll=False)
                saveToFile("\nDECISION:", printAll=False)
                saveToFile(torch.round(pred), printAll=False)
                saveToFile("\nGROUND TRUTH:", printAll=False)
                saveToFile(ground_truth, printAll=False)
                saveToFile("------------------------------------------------------------------------", printAll=False)
                saveToFile("\n", printAll=False)

            # Free the cache
            data.x = data.x.cpu()
            data.x = old_data
            pred = pred.cpu()
            ground_truth = ground_truth.cpu()
            del pred, ground_truth, result, loss

        batch_loss = torch.div(singular_loss, len(batch))
        total_loss += batch_loss.item()
        total_accuracy += singular_accuracy.item()

        # Compute precision, recall, and f1 for the batch
        batch_pred = torch.cat(batch_pred)
        batch_ground_truth = torch.cat(batch_ground_truth)
        precision.update(batch_pred, batch_ground_truth)
        recall.update(batch_pred, batch_ground_truth)
        f1.update(batch_pred, batch_ground_truth)

    # Compute precision, recall, and f1 for the entire validation set
    precision_result = precision.compute()
    recall_result = recall.compute()
    f1_result = f1.compute()

    return total_accuracy, total_loss, precision_result.item(), recall_result.item(), f1_result.item()




def test(test_set):
	""" Test the model """
	
	saveToFile = SaveOutput('./test_log/', LOG_NAME, debug=DEBUG)
	data_list, ground_truth_list, node_shapes, mapping_list = create_graphs(test_set)
	min_num_nodes = 0
	batch_size = 0
	for i, data in enumerate(data_list):
		if min_num_nodes + data.x.shape[0] <= MAX_NUM_NODES:
			min_num_nodes += data.x.shape[0]
			batch_size += 1
	
	# Load model
	model = GNN3(input_dim=768)
	if(LOAD_MODEL_PATH != ""):model.load(LOAD_MODEL_PATH)
	model = model.to(model_device)
	model.eval()


	total_accuracy = 0
	precision = Precision(task='binary').to(model_device)
	recall = Recall(task='binary').to(model_device)
	f1 = F1Score(task='binary').to(model_device)

	for i in tqdm(range(0, len(data_list), batch_size), desc="Batch"):  # For each batch

		singular_accuracy = 0
		batch_pred = []
		batch_ground_truth = []

		# get the batch
		batch = data_list[i:i+batch_size]
		shapes = node_shapes[i:i+batch_size]
		ground_truth_batch = ground_truth_list[i:i+batch_size]
		mapping_list_batch = mapping_list[i:i+batch_size]


		for index, data in enumerate(batch):  # For each graph
			# generate the BERT embeddings
			old_data = data.x.clone()
			data.x = data.x.to(model_device)
			data.x = embeddings_generator.create_characterBERT_embeddings(data.x[:shapes[index]])

			# Pad the nodes to the maximum dimension with zeros
			if shapes[index] < MAX_DIM_GRAPH:
				null_nodes = torch.zeros(size=((MAX_DIM_GRAPH - shapes[index]), data.x.shape[1]))
				null_nodes = null_nodes.to(model_device)
				data.x = torch.cat((data.x[:shapes[index]], null_nodes), dim=0)

			# Forward pass
			data.to(model_device)
			pred = model.forward(data)
			ground_truth = ground_truth_batch[index].to(model_device)

			# Compute accuracy
			result = torch.eq(torch.round(pred), ground_truth)
			singular_accuracy = torch.add(singular_accuracy, (torch.sum(result) / pred.numel()))
			batch_pred.append(torch.round(pred))
			batch_ground_truth.append(ground_truth)

			if i >= len(data_list) - len(batch) and index == len(batch) - 1:
				saveToFile("DECISION FOR THE LAST GRAPH IN THE BATCH IN THE VALIDATION STEP -----------------------------------", printAll=False)
				saveToFile("GRAPH: " + str(index), printAll=False)
				saveToFile("MAPPING_LIST: " + str(mapping_list_batch[index]), printAll=False)
				saveToFile("", printAll=False)
				saveToFile("PREDICTION:", printAll=False)
				saveToFile(pred, printAll=False)
				saveToFile("\nDECISION:", printAll=False)
				saveToFile(torch.round(pred), printAll=False)
				saveToFile("\nGROUND TRUTH:", printAll=False)
				saveToFile(ground_truth, printAll=False)
				saveToFile("------------------------------------------------------------------------", printAll=False)
				saveToFile("\n", printAll=False)

			# Free the cache
			data.x = data.x.cpu()
			data.x = old_data
			pred = pred.cpu()
			ground_truth = ground_truth.cpu()
			del pred, ground_truth, result, loss


		# Compute precision, recall, and f1 for the batch
		total_accuracy += singular_accuracy.item()
		batch_pred = torch.cat(batch_pred)
		batch_ground_truth = torch.cat(batch_ground_truth)
		precision.update(batch_pred, batch_ground_truth)
		recall.update(batch_pred, batch_ground_truth)
		f1.update(batch_pred, batch_ground_truth)


	# Compute precision, recall, and f1 for the entire validation set
	total_accuracy /= len(data_list)
	precision_result = precision.compute().item()
	recall_result = recall.compute().item()
	f1_result = f1.compute().item()
 

	saveToFile("Testing performance:")
	saveToFile("Accuracy: " + str(total_accuracy))
	saveToFile("Precision: " + str(precision_result))
	saveToFile("Recall: " + str(recall_result))
	saveToFile("F1: " + str(f1_result))
	saveToFile("")







def split_dataset():
	""" Split the dataset into train and test datasets """

	saveToFile = SaveOutput('./log/', LOG_NAME, debug=DEBUG)
	train_datasets_exists = exists(TRAIN_PATH)
	valid_datasets_exists = exists(VALID_PATH)
	test_datasets_exists = exists(TEST_PATH)	

	if FORCE_DATASET_SPLIT:
		train_datasets_exists = False
		valid_datasets_exists = False
		test_datasets_exists = False
	
 
	if train_datasets_exists and test_datasets_exists and valid_datasets_exists:
		saveToFile("Dataset already splitted!\nLoading training and testing dataset...")
		train_df = pd.read_csv(TRAIN_PATH)
		test_df = pd.read_csv(TEST_PATH)
		valid_df = pd.read_csv(VALID_PATH)
	else:
		saveToFile("\nDataset not found!. Create train and test...")
		
		# Splitting in training and testing set mantaining the same proportion of the original IsShared column in the splitted dataset 
		dataset = pd.read_csv(DATASET_PATH)
		dataset = dataset.drop("Unnamed: 0", axis=1)
		iban_list = dataset.AccountNumber.unique()

  
		isShared = dataset.groupby('AccountNumber', sort=False)['IsShared'].first().loc[iban_list].values
		train_iban_list, test_iban_list = train_test_split(iban_list, train_size=TRAIN_SIZE, stratify=isShared)
		train_df = dataset.loc[dataset.AccountNumber.isin(train_iban_list)]
		test_df = dataset.loc[dataset.AccountNumber.isin(test_iban_list)]

		isShared = test_df.groupby('AccountNumber', sort=False)['IsShared'].first().loc[test_iban_list].values
		test_iban_list, valid_iban_list = train_test_split(test_iban_list, train_size=VALID_SIZE, stratify=isShared)
		valid_df = test_df.loc[test_df.AccountNumber.isin(valid_iban_list)]
		test_df = test_df.loc[test_df.AccountNumber.isin(test_iban_list)]

		train_df.to_csv(TRAIN_PATH)
		test_df.to_csv(TEST_PATH)
		valid_df.to_csv(VALID_PATH)


	return dataset, train_df, test_df, valid_df


	
def main():
	""" Main function: 
 		- Split the dataset into train and test datasets
 		- Train the model
 		- Test the model
	"""
	
	saveToFile = SaveOutput('./log/', LOG_NAME, debug=DEBUG)
	dataset, train_df, test_df, valid_df = split_dataset()
	if train_df.shape[0] + test_df.shape[0] + valid_df.shape[0] != dataset.shape[0]:saveToFile("Dataset not correctly splitted!")
	else: saveToFile("Dataset correctly splitted!")
	saveToFile("Datasets loaded!\n") 
	
	if TRAIN:
		model = train(train_df, valid_df)
		if (SAVE_MODEL): model.save(MODEL_PATH)
	else:test(test_df)



if __name__ == "__main__":
	main()
