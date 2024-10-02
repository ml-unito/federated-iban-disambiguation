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

with open('./config/parameters.json', "r") as data_file:
	parameters = json.load(data_file)

DATASET_PATH = parameters["dataset_path"]
TRAIN_SIZE = parameters["train_size"]
TRAIN_PATH = parameters["train_path"]
TEST_PATH = parameters["test_path"]
TRAIN = parameters["train"]
MAX_DIM_GRAPH = parameters["max_dim_graph"]


def union_ground_truth(ground_truth_list):
	complete_ground_truth = None
	for index, ground_truth in enumerate(ground_truth_list):
		if index == 0:
			complete_ground_truth = ground_truth
		else:
			complete_ground_truth = torch.cat(
				(torch.cat((complete_ground_truth, torch.zeros(MAX_DIM_GRAPH * index, MAX_DIM_GRAPH)), dim=1),
				 torch.cat((torch.zeros(MAX_DIM_GRAPH, MAX_DIM_GRAPH * index), ground_truth), dim=1)),
				dim=0)

	return complete_ground_truth


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
	mapping = {index: i for i, index in enumerate(group.index.unique())}

	nodes_list = []
	for elem in group[field]:
		node_embedding = embeddings_generator.create_characterBERT_embeddings(elem)
		nodes_list.append(node_embedding)
	x = torch.cat(nodes_list, 0)

	if x.shape[0] < MAX_DIM_GRAPH:
		null_nodes = torch.zeros(size=((MAX_DIM_GRAPH - x.shape[0]), x.shape[1]))
		x = torch.cat((x, null_nodes), dim=0)

	return x, mapping


def create_graphs(dataset):
	data_list = []
	ground_truth_list = []
	for iban, group in tqdm(dataset.groupby(["AccountNumber"], sort=False), desc="Creating graph"):
		if len(group) > MAX_DIM_GRAPH:
			print("The number of nodes (" + str(len(group)) + ") is greater than the limit (" + str(
				MAX_DIM_GRAPH) + "). The entries number was reduced.")
			group = group[:MAX_DIM_GRAPH]

		node, node_mapping = generate_nodes(group, field="Name")
		edge_index = generate_edges(node_mapping)

		data = Data(x=node, edge_index=edge_index.t().contiguous())
		data.validate(raise_on_error=True)
		data_list.append(data)

		ground_truth = generate_ground_truth(group, node_mapping)
		ground_truth_list.append(ground_truth)

	return data_list, ground_truth_list


def test(dataset):
	pass


def train(dataset):
	# f = open("log.txt", "w")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: '{device}'")
	torch.set_printoptions(profile="full", linewidth=1000)
	data_list, ground_truth_list = create_graphs(dataset)

	data_batch = Batch.from_data_list(data_list).to(device)
	ground_truth = union_ground_truth(ground_truth_list).to(device)

	node_features_dim = data_list[0].num_node_features
	model = GNN(input_dim=node_features_dim).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
	for epoch in tqdm(range(1, 5), desc="Training"):
		optimizer.zero_grad()
		pred = model.forward(data_batch.to(device))
		# print("pred\n", pred)
		# f.write("PRED\n")
		# f.write(str(pred))

		ground_truth = ground_truth.to(device)
		# print("ground_truth\n", ground_truth.shape, ground_truth)
		# f.write("GROUND TRUTH\n")
		# f.write(str(ground_truth))

		loss = torch.norm(input=torch.sub(pred, ground_truth), p="fro")
		loss.backward(retain_graph=True)
		optimizer.step()

		accuracy = (torch.sum(torch.eq(pred, ground_truth)) / pred.numel())

		print("\n====== epoch " + str(epoch) + " ======")
		print("loss:", loss.item())
		print("accuracy:", accuracy.item())

	# f.close()


def split_dataset():
	train_datasets_exists = exists(TRAIN_PATH)
	test_datasets_exists = exists(TEST_PATH)

	if train_datasets_exists and test_datasets_exists:
		print("Load train and test dataset")
		train_df = pd.read_csv(TRAIN_PATH, index_col="Index")
		test_df = pd.read_csv(TEST_PATH, index_col="Index")
	else:
		print("Create train and test dataset")
		dataset = pd.read_csv(DATASET_PATH, index_col="Index")

		iban_list = dataset.AccountNumber.unique()
		train_iban_list, test_iban_list = train_test_split(iban_list, train_size=TRAIN_SIZE)

		train_df = dataset.loc[dataset.AccountNumber.isin(train_iban_list)]
		train_df.to_csv(TRAIN_PATH)

		test_df = dataset.loc[dataset.AccountNumber.isin(test_iban_list)]
		test_df.to_csv(TEST_PATH)

	return train_df, test_df


def main():
	train_df, test_df = split_dataset()
	if TRAIN:
		train(train_df)
	else:
		test(test_df)


if __name__ == "__main__":
	main()
