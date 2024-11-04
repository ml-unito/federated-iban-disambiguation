import json
import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from os.path import exists
from tqdm import tqdm
from itertools import combinations
from gnn_model import GNN, GNN2
import embedding_generator.main.embeddings_generator as embeddings_generator
import itertools
from sklearn.metrics import classification_report
import pickle
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

with open('./config/parameters.json', "r") as data_file:
	parameters = json.load(data_file)

DATASET_DIR = parameters["dataset_directory"]
DATASET_NAME = parameters["dataset_name"]
TRAIN_SIZE = parameters["train_size"]
VALIDATION_SIZE = parameters["validation_size"]
TRAIN_PATH = parameters["train_path"]
TEST_PATH = parameters["test_path"]
VALIDATION_PATH = parameters["validation_path"]
TRAIN = parameters["train"]
MAX_DIM_GRAPH = parameters["max_dim_graph"]
NUM_EPOCH = parameters["num_epoch"]
LEARNING_RATE = parameters["learning_rate"]
MODEL_PATH = "./model/model_weights.pth"


def plot_metrics(training_loss_list, training_accuracy_list, validation_loss_list, validation_accuracy_list):
	plt.figure(figsize=(10, 5))

	plt.subplot(1, 2, 1)
	plt.plot(range(1, NUM_EPOCH + 1), training_loss_list, 'b', label="training loss")
	plt.plot(range(1, NUM_EPOCH + 1), validation_loss_list, 'r', label="validation loss")
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(range(1, NUM_EPOCH + 1), training_accuracy_list, 'b', label="training accuracy")
	plt.plot(range(1, NUM_EPOCH + 1), validation_accuracy_list, 'r', label="validation accuracy")
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.tight_layout(w_pad=4)

	plt.savefig("./plot/plot_loss" + str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + ".png")


def test(dataset):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	data_list, ground_truth_list = create_graphs(dataset)

	node_features_dim = data_list[0].num_node_features
	model = GNN(input_dim=node_features_dim).to(device)
	model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
	model.eval()

	partial_accuracy = 0
	ground_truth_l = []
	pred_l = []
	for index, data in enumerate(data_list):
		# print(index)
		pred = model.forward(data.to(device))
		ground_truth = ground_truth_list[index].to(device)
		# print("ground_truth",ground_truth)

		# Removes padding nodes from result
		pred = pred[:ground_truth.shape[0], :ground_truth.shape[1]]
		# print("pred",pred)
		rounded_pred = torch.where(pred > 0.5, 1, 0)

		accuracy = (torch.sum(torch.eq(rounded_pred, ground_truth)) / rounded_pred.numel())
		partial_accuracy = torch.add(partial_accuracy, accuracy)

		ground_truth_l.append(ground_truth.flatten())
		pred_l.append(rounded_pred.flatten())

	ground_truth_l = torch.cat(ground_truth_l, dim=-1)
	pred_l = torch.cat(pred_l, dim=-1)
	report = classification_report(ground_truth_l.cpu().detach().numpy(), pred_l.cpu().detach().numpy())
	print(report)

	avg_accuracy = torch.div(partial_accuracy, len(data_list)).item()


# print("accuracy:", avg_accuracy)


def training_step(model, optimizer, device, epoch, train_data_list, train_ground_truth_list, f):
	model.train()

	partial_accuracy = 0
	total_loss = total_examples = 0

	for index, data in enumerate(train_data_list):
		optimizer.zero_grad()
		pred = model.forward(data.to(device))
		# print("pred",pred)

		ground_truth = train_ground_truth_list[index].to(device)
		# print("groud",ground_truth)
		# f.write("\n\GROUND TRUTH\n")
		# f.write(str(ground_truth))

		# Removes padding nodes from result
		pred = pred[:ground_truth.shape[0], :ground_truth.shape[1]]
		# print("pred",pred)
		# f.write("\n\nPRED\n")
		# f.write(str(pred))

		rounded_pred = torch.where(pred > 0.5, 1, 0)
		# print("rounded_pred",rounded_pred)
		# f.write("\n\nrounded PRED\n")
		# f.write(str(rounded_pred))

		# loss_value = loss(pred, ground_truth)
		loss_value = F.binary_cross_entropy_with_logits(pred, ground_truth)
		loss_value.backward()
		optimizer.step()

		# partial_loss = torch.add(partial_loss, loss_value)
		total_loss += float(loss_value) * pred.numel()
		total_examples += pred.numel()

		accuracy = (torch.sum(torch.eq(rounded_pred, ground_truth)) / rounded_pred.numel())
		partial_accuracy = torch.add(partial_accuracy, accuracy)

	# avg_loss = torch.div(partial_loss, len(train_data_list))
	avg_loss = total_loss / total_examples
	avg_accuracy = torch.div(partial_accuracy, len(train_data_list))

	# print("\n====== epoch " + str(epoch) + " ======")
	# print("loss:", avg_loss)
	# print("accuracy:", avg_accuracy.item())

	f.write("\n====== epoch " + str(epoch) + " ======\n")
	f.write("loss:\t" + str(avg_loss) + "\n")
	f.write("accuracy:\t" + str(avg_accuracy.item()) + "\n")

	return avg_loss, avg_accuracy.item()


def validation_step(model, device, epoch, data_list, ground_truth_list, f):
	model.eval()

	partial_accuracy = 0
	total_loss = total_examples = 0

	for index, data in enumerate(data_list):
		pred = model.forward(data.to(device))
		# print("pred",pred)

		ground_truth = ground_truth_list[index].to(device)
		# print("groud",ground_truth)
		# f.write("\n\GROUND TRUTH\n")
		# f.write(str(ground_truth))

		# Removes padding nodes from result
		pred = pred[:ground_truth.shape[0], :ground_truth.shape[1]]
		# print("pred",pred)
		# f.write("\n\nPRED\n")
		# f.write(str(pred))

		rounded_pred = torch.where(pred > 0.5, 1, 0)
		# print("rounded_pred",rounded_pred)
		# f.write("\n\nrounded PRED\n")
		# f.write(str(rounded_pred))

		# loss_value = loss(pred, ground_truth)
		loss_value = F.binary_cross_entropy_with_logits(pred, ground_truth)

		# partial_loss = torch.add(partial_loss, loss_value)
		total_loss += float(loss_value) * pred.numel()
		total_examples += pred.numel()

		accuracy = (torch.sum(torch.eq(rounded_pred, ground_truth)) / rounded_pred.numel())
		partial_accuracy = torch.add(partial_accuracy, accuracy)

	# avg_loss = torch.div(partial_loss, len(train_data_list))
	avg_loss = total_loss / total_examples
	avg_accuracy = torch.div(partial_accuracy, len(data_list))

	# print("\n====== epoch " + str(epoch) + " ======")
	# print("loss:", avg_loss)
	# print("accuracy:", avg_accuracy.item())

	f.write("\n====== valid epoch " + str(epoch) + " ======\n")
	f.write("loss:\t" + str(avg_loss) + "\n")
	f.write("accuracy:\t" + str(avg_accuracy.item()) + "\n")

	return avg_loss, avg_accuracy.item()


def train(train_df, validation_df):
	f = open_log_file()
	# torch.set_printoptions(profile="full", linewidth=300)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# print(f"Device: '{device}'")

	train_data_list = None
	train_ground_truth_list = None
	validation_data_list = None
	validation_ground_truth_list = None

	file_name_training_graphs = "training_graphs_" + DATASET_NAME[:-4] + ".pkl"
	file_name_validation_graphs = "validation_graphs_" + DATASET_NAME[:-4] + ".pkl"
	if exists(file_name_training_graphs) and exists(file_name_validation_graphs):
		print("Load graphs from file")
		train_data_list, train_ground_truth_list = load_graphs_from_file(file_name=file_name_training_graphs)
		validation_data_list, validation_ground_truth_list = load_graphs_from_file(file_name=file_name_validation_graphs)
	else:
		train_data_list, train_ground_truth_list = create_graphs(train_df, to_save=True,
																														 file_name=file_name_training_graphs,
																														 to_generate_correct_edge=True)
		validation_data_list, validation_ground_truth_list = create_graphs(validation_df, to_save=True,
																																			 file_name=file_name_validation_graphs)
		exit()

	node_features_dim = train_data_list[0].num_node_features
	model = GNN(input_dim=node_features_dim).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	f.write("learning rate: " + str(LEARNING_RATE) + "\n")
	f.write("loss: binary_cross_entropy_with_logits\n")
	f.write("train set: " + str(len(train_data_list)) + " graphs\n")

	training_loss_list = []
	training_accuracy_list = []

	validation_loss_list = []
	validation_accuracy_list = []

	for epoch in tqdm(range(1, NUM_EPOCH + 1), desc="Training"):
		train_loss, train_accuracy = training_step(model, optimizer, device, epoch, train_data_list,
																							 train_ground_truth_list, f)
		training_loss_list.append(train_loss)
		training_accuracy_list.append(train_accuracy)

		validation_loss, validation_accuracy = validation_step(model, device, epoch, validation_data_list,
																													 validation_ground_truth_list, f)
		validation_loss_list.append(validation_loss)
		validation_accuracy_list.append(validation_accuracy)

	plot_metrics(training_loss_list, training_accuracy_list, validation_loss_list, validation_accuracy_list)

	torch.save(model.state_dict(), MODEL_PATH)

	f.close()


def generate_ground_truth(dataset_group, node_mapping):
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
	# if num_valued_nodes < MAX_DIM_GRAPH:
	# 	num_empty_nodes = MAX_DIM_GRAPH - num_valued_nodes
	#
	# 	top_right_tensors_padding_nodes = torch.zeros((num_valued_nodes, num_empty_nodes))
	# 	bottom_tensors_padding_nodes = torch.cat(
	# 		(torch.zeros(num_empty_nodes, num_valued_nodes), torch.ones(num_empty_nodes, num_empty_nodes)),
	# 		dim=1)
	#
	# 	ground_truth = torch.cat(
	# 		(torch.cat((ground_truth, top_right_tensors_padding_nodes), dim=1), bottom_tensors_padding_nodes),
	# 		dim=0)

	return ground_truth


def generate_edges(dataset_group, node_mapping, to_generate_correct_edge):
	"""It generates the edges between all possible pairs of entries and between
	all possible pairs of padding nodes."""

	if to_generate_correct_edge:
		edges = []
		for node1 in node_mapping.keys():
			for node2 in node_mapping.keys():
				if node1 == node2 or dataset_group.loc[node1]["Holder"] == dataset_group.loc[node2]["Holder"]:
					edges.append([node_mapping[node1], node_mapping[node2]])

		edge_index = torch.tensor(edges, dtype=torch.long)
	else:
		index_nodes = list(node_mapping.values())
		edges = [[couple[0], couple[1]] for couple in list(combinations(index_nodes, 2))]

		edges += [[index, index] for index in node_mapping.values()]

		index_padding_nodes = list(range(len(node_mapping.values()), MAX_DIM_GRAPH))
		edges_padding_nodes = [[couple[0], couple[1]] for couple in list(combinations(index_padding_nodes, 2))]

		complete_edges = edges + edges_padding_nodes
		edge_index = torch.tensor(complete_edges, dtype=torch.long)
	# edge_index = torch.tensor(edges, dtype=torch.long)

	# Creazione archi solo tra nodi uguali
	# comb = list(combinations(list(node_mapping.keys()), 2))
	# for pair in comb:
	# 	if dataset_group.loc[pair[0]]["Name"] == dataset_group.loc[pair[1]]["Name"]:
	# 		edges.append([node_mapping[pair[0]], node_mapping[pair[1]]])

	# edges += [[index, index] for index in node_mapping.values()]
	# edge_index = torch.tensor(edges, dtype=torch.long)

	return edge_index


def generate_nodes(dataset, field):
	mapping = {index: i for i, index in enumerate(dataset.index)}

	nodes_list = []
	for elem in dataset[field]:
		node_embedding = embeddings_generator.create_characterBERT_embeddings(elem)
		nodes_list.append(node_embedding)
	x = torch.cat(nodes_list, 0)

	if x.shape[0] < MAX_DIM_GRAPH:
		null_nodes = torch.zeros(size=((MAX_DIM_GRAPH - x.shape[0]), x.shape[1]))
		x = torch.cat((x, null_nodes), dim=0)

	return x, mapping


def save_graphs_into_file(data_list, ground_truth_list, file_name):
	data = {
		"data_list": data_list,
		"ground_truth_list": ground_truth_list
	}

	file = open(file_name, 'wb')
	pickle.dump(data, file)
	file.close()


def load_graphs_from_file(file_name):
	file = open(file_name, 'rb')
	data = pickle.load(file)
	file.close()

	return data["data_list"], data["ground_truth_list"]


def create_graphs(dataset, to_save=False, file_name=None, to_generate_correct_edge=False):
	data_list = []
	ground_truth_list = []

	for iban, group in tqdm((dataset.groupby(["AccountNumber"], sort=False)), desc="Creating graph"):
		if len(group) > MAX_DIM_GRAPH:
			print("The number of nodes (" + str(len(group)) + ") is greater than the limit (" + str(
				MAX_DIM_GRAPH) + "). The entries number was reduced.")
			group = group[:MAX_DIM_GRAPH]

		node, node_mapping = generate_nodes(group, field="Name")
		edge_index = generate_edges(group, node_mapping, to_generate_correct_edge)

		data = Data(x=node, edge_index=edge_index.t().contiguous())
		data.validate(raise_on_error=True)
		data_list.append(data)

		ground_truth = generate_ground_truth(group, node_mapping)
		ground_truth_list.append(ground_truth)

	if to_save:
		save_graphs_into_file(data_list, ground_truth_list, file_name)

	return data_list, ground_truth_list


def open_log_file():
	now = datetime.now()
	return open("./log/log_graphs_" + str(now.strftime("%d-%m-%Y_%H-%M-%S")) + ".txt", "w")


def split_dataset():
	""" Split the dataset into train and test datasets """

	train_datasets_exists = exists(TRAIN_PATH)
	valid_datasets_exists = exists(VALIDATION_PATH)
	test_datasets_exists = exists(TEST_PATH)

	if train_datasets_exists and test_datasets_exists and valid_datasets_exists:
		train_df = pd.read_csv(TRAIN_PATH, index_col="Index")
		test_df = pd.read_csv(TEST_PATH, index_col="Index")
		valid_df = pd.read_csv(VALIDATION_PATH, index_col="Index")
	else:
		# Splitting in training and testing set mantaining the same proportion of the original IsShared column in the splitted dataset
		dataset = pd.read_csv(DATASET_DIR + DATASET_NAME)
		iban_list = dataset.AccountNumber.unique()

		isShared = dataset.groupby('AccountNumber', sort=False)['IsShared'].first().loc[iban_list].values
		train_iban_list, test_iban_list = train_test_split(iban_list, train_size=TRAIN_SIZE, stratify=isShared)
		train_df = dataset.loc[dataset.AccountNumber.isin(train_iban_list)]
		test_df = dataset.loc[dataset.AccountNumber.isin(test_iban_list)]

		isShared = test_df.groupby('AccountNumber', sort=False)['IsShared'].first().loc[test_iban_list].values
		test_iban_list, valid_iban_list = train_test_split(test_iban_list, train_size=VALIDATION_SIZE, stratify=isShared)
		valid_df = test_df.loc[test_df.AccountNumber.isin(valid_iban_list)]
		test_df = test_df.loc[test_df.AccountNumber.isin(test_iban_list)]

		train_df.to_csv(TRAIN_PATH)
		test_df.to_csv(TEST_PATH)
		valid_df.to_csv(VALIDATION_PATH)

	return train_df, test_df, valid_df


def main():
	train_df, test_df, validation_df = split_dataset()
	if TRAIN:
		train(train_df, validation_df)
	else:
		test(test_df)


if __name__ == "__main__":
	main()
