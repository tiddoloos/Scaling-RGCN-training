import torch
from torch import nn
from torch import functional as F
from torch_geometric.nn import RGCNConv

class RGCNLayer(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_l, num_classes):
        super(RGCNLayer, self).__init__()
        self.rgcn1 = RGCNConv(num_nodes, hidden_l, num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_classes, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.conv1(None, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = torch.nn.Sigmoid(x)
        return x