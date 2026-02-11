import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GNN(torch.nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.gatconv1 = GATConv(in_channels=input_dim, out_channels=input_dim)

	def forward(self, graph):
		x = graph.x
		edge_index = graph.edge_index

		x = self.gatconv1(x, edge_index)
		x = F.leaky_relu(x, negative_slope=0.02)
		x = F.dropout(x, p=0.2, training=self.training)

		x = x / torch.norm(x)
		xT = x.t()
		pred = x @ xT

		return torch.sigmoid(pred)


class GNN2(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super().__init__()
		self.gatconv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim)
		self.gatconv2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)

	def forward(self, graph):
		x = graph.x
		edge_index = graph.edge_index

		x = self.gatconv1(x, edge_index)
		x = F.leaky_relu(x, negative_slope=0.02)
		x = F.dropout(x, p=0.2, training=self.training)

		x = self.gatconv2(x, edge_index)
		x = F.leaky_relu(x, negative_slope=0.02)
		x = F.dropout(x, p=0.2, training=self.training)

		xT = x.t()
		pred = x @ xT

		return pred