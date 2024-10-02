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

        residual = x
        x = self.gatconv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.02)
        x = x + residual
        x = F.dropout(x, p=0.2, training=self.training)

        x = x / torch.norm(x)
        xT = x.t()
        pred = x @ xT

        return pred



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


class GNN3(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gatconv1 = GATConv(in_channels=input_dim, out_channels=256)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index

        x = self.gatconv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.02)
        x = F.dropout(x, p=0.8, training=self.training)

        x = x / torch.norm(x)
        xT = x.t()
        pred = x @ xT
        
        return torch.sigmoid(pred)





# class GNN3(torch.nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.gatconv1 = GATConv(in_channels=input_dim, out_channels=128)
#         self.gatconv2 = GATConv(in_channels=128, out_channels=128)
#         self.gatconv3 = GATConv(in_channels=128, out_channels=128)


#     def forward(self, graph):
#         x = graph.x
#         edge_index = graph.edge_index

#         x = self.gatconv1(x, edge_index)
#         x = F.leaky_relu(x, negative_slope=0.05)
#         x = F.dropout(x, p=0.6, training=self.training)
        
#         # x = self.gatconv2(x, edge_index)
#         # x = F.leaky_relu(x, negative_slope=0.02)
#         # x = F.dropout(x, p=0.8, training=self.training)
        
#         # x = self.gatconv3(x, edge_index)
#         # x = F.leaky_relu(x, negative_slope=0.02)
#         # x = F.dropout(x, p=0.8, training=self.training)

#         x = x / torch.norm(x)
#         xT = x.t()
#         pred = x @ xT

#         # return torch.tanh(pred)
#         # Apply the classifier to the predicted adjacency matrix
#         return torch.sigmoid(pred)
#         #return torch.sigmoid(pred.view(-1)).view(pred.shape)


