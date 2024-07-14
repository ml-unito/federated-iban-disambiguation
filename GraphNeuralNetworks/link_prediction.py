import json
import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from torch import Tensor
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from os.path import exists
from tqdm import tqdm
from itertools import combinations
from gnn_model import GNN


with open('./config/parameters.json', "r") as data_file:
	parameters = json.load(data_file)

DATASET_PATH = parameters["dataset_path"]
TRAIN_SIZE = parameters["train_size"]
TRAIN_PATH = parameters["train_path"]
TEST_PATH = parameters["test_path"]
TRAIN = parameters["train"]


# TODO: da sostituire con CharacterBERT
class SequenceEncoder:
	def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
		self.device = device
		self.model = SentenceTransformer(model_name, device=device)

	@torch.no_grad()
	def __call__(self, df):
		x = self.model.encode(df.values, show_progress_bar=False, convert_to_tensor=True, device=self.device)
		return x.cpu()


def test(dataset):
	pass


def train(dataset):
	data_list, ground_truth_list = create_graphs(dataset)
	print(data_list)
	# data_batch = Batch.from_data_list(data_list)
	# print(data_batch)
	# print("Batch size " + str(data_batch.num_graphs))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# print(f"Device: '{device}'")

	node_features_dim = data_list[0].num_node_features
	model = GNN(input_dim=node_features_dim)
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	for epoch in tqdm(range(1, 6), desc="Training"):
		optimizer.zero_grad()
		partial_loss = 0

		for index, data in enumerate(data_list):
			# print(data)
			pred = model.forward(data.to(device))
			# print(pred)
			ground_truth = ground_truth_list[index].to(device)
			# print(ground_truth)

			loss = torch.norm(input=torch.sub(pred, ground_truth), p="fro")
			#print(loss)

			partial_loss = torch.add(partial_loss, loss)

		total_loss = torch.div(partial_loss, len(data_list))

		total_loss.backward()
		optimizer.step()
		print(total_loss)


def create_graphs(dataset):
	data_list = []
	correct_edges = []

	for iban, group in tqdm(dataset.groupby(["AccountNumber"]), desc="Creating graph"):
		node, node_mapping = generate_node(group, encoders={'Name': SequenceEncoder()})
		edge_index = generate_edges(node_mapping)

		data = Data(x=node, edge_index=edge_index.t().contiguous())
		data.validate(raise_on_error=True)
		data_list.append(data)

		is_shared = group["IsShared"].tolist()[0]

		if not is_shared:
			ground_truth = torch.ones(data.num_nodes, data.num_nodes) - torch.diag(torch.tensor([1] * data.num_nodes))
		else:
			ground_truth = torch.zeros(data.num_nodes, data.num_nodes)
			for node1 in node_mapping.keys():
				for node2 in node_mapping.keys():
					if node1 != node2 and group.loc[node1]["Holder"] == group.loc[node2]["Holder"]:
						ground_truth[node_mapping[node1], node_mapping[node2]] = 1

		correct_edges.append(ground_truth)

	return data_list, correct_edges


def generate_edges(node_mapping):
	"""It generates the edges between all possible pairs of nodes."""

	edges = [[couple[0], couple[1]] for couple in list(combinations(list(node_mapping.values()), 2))]
	edge_index = torch.tensor(edges, dtype=torch.long)

	return edge_index


def generate_node(dataset, encoders=None):
	mapping = {index: i for i, index in enumerate(dataset.index.unique())}

	x = None
	if encoders is not None:
		xs = [encoder(dataset[col]) for col, encoder in encoders.items()]
		x = torch.cat(xs, dim=-1)

	return x, mapping


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
