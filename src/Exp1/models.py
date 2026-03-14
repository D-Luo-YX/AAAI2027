import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, model_name="gcn", dropout=0.5):
        super().__init__()

        if model_name == "gcn":
            conv = GCNConv
        elif model_name == "sage":
            conv = SAGEConv
        else:
            raise ValueError("model_name must be 'gcn' or 'sage'")

        self.conv1 = conv(in_dim, hidden_dim)
        self.conv2 = conv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
