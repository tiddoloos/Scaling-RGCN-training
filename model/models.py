import torch
import torch.nn.functional as F
from typing import List
from torch import Tensor
from torch import nn
from torch_geometric.nn import RGCNConv
from graphdata.graphData import Graph

class base_Layers(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_l: int, num_labels: int) -> None:
        super(base_Layers, self).__init__()
        self.rgcn1 = RGCNConv(in_channels=num_nodes, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, graph: Graph)-> Tensor:
        x = self.rgcn1(None, graph.edge_index, graph.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, graph.edge_index, graph.edge_type)
        x = torch.sigmoid(x)
        return x

###############################################################################################################
class emb_layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, emb_dim) -> None:
        super(emb_layers, self).__init__()
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, graph: Graph) -> Tensor:
        x = self.rgcn1(graph.embedding.weight, graph.edge_index, graph.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, graph.edge_index, graph.edge_type)
        x = torch.sigmoid(x)
        return x
    
    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)
    
##############################################################################################################

class emb_mlp_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, in_f: int, out_f: int, emb_dim):
        super(emb_mlp_Layers, self).__init__()
        self.lin1 = nn.Linear(in_features=in_f, out_features=out_f)
        self.lin2 = nn.Linear(in_features=out_f, out_features=emb_dim)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, graph: Graph):
        x = torch.tanh(self.lin1(graph.embedding.weight))
        x = self.lin2(x)
        embedding = nn.Embedding.from_pretrained(x, freeze=True)
        x = self.rgcn1(embedding.weight, graph.edge_index, graph.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, graph.edge_index, graph.edge_type)
        x = torch.sigmoid(x)
        return x
    
    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)

######################################################################################################################

class emb_att_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, num_sums: int, emb_dim):
        super(emb_att_Layers, self).__init__()
        self.embedding = None
        self.att = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_sums)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, graph: Graph) -> Tensor:
        attn_output, attn_output_weights = self.att(graph.embedding, graph.embedding, graph.embedding)
        x = self.rgcn1(attn_output[0], graph.edge_index, graph.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, graph.edge_index, graph.edge_type)
        x = torch.sigmoid(x)
        return x

    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)
