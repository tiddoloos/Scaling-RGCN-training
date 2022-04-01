import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_l, num_labels):
        super(RGCN, self).__init__()
        self.rgcn1 = RGCNConv(num_nodes, hidden_l, num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.rgcn1(None, edge_index, edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = torch.sigmoid(x)
        return x