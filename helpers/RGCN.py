import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_l, num_labels):
        super(RGCN, self).__init__()
        self.rgcn1 = RGCNConv(num_nodes, hidden_l, num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_out')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_out')

    def forward(self, edge_index, edge_type):
        x = self.rgcn1(None, edge_index, edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = torch.sigmoid(x)
        return x
    
    def override_params(self, weight_1, bias_1, root_1, weight_2, bias_2, root_2):
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)

    def reset_weights(self):
        for layer in self.children():
            print(layer)
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()