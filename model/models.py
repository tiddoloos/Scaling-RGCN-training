import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data 


class BaseLayers(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_l: int, num_labels: int) -> None:
        super(BaseLayers, self).__init__()
        #in_channels => embedding size
        self.rgcn1 = RGCNConv(in_channels=num_nodes, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, training_data: Data)-> Tensor:
        x = self.rgcn1(None, training_data.edge_index, training_data.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, training_data.edge_index, training_data.edge_type)
        x = torch.sigmoid(x)
        return x


class Emb_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, emb_dim: int) -> None:
        super(Emb_Layers, self).__init__()
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, training_data: Data) -> Tensor:
        x = self.rgcn1(training_data.embedding.weight, training_data.edge_index, training_data.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, training_data.edge_index, training_data.edge_type)
        x = torch.sigmoid(x)
        return x
    
    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)


class Emb_ATT_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, emb_dim: int) -> None:
        super(Emb_ATT_Layers, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=3)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, training_data: Data) -> Tensor:
        attn_output, _ = self.att(training_data.embedding, training_data.embedding, training_data.embedding)
        x = self.rgcn1(attn_output[0], training_data.edge_index, training_data.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, training_data.edge_index, training_data.edge_type)
        x = torch.sigmoid(x)
        return x

    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)


class Emb_MLP_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, emb_dim: int):
        in_f = 3*emb_dim
        out_f = round((in_f/2)*3 + num_labels)
        super(Emb_MLP_Layers, self).__init__()
        self.lin1 = nn.Linear(in_features= in_f, out_features=out_f)
        self.lin2 = nn.Linear(in_features=out_f, out_features=emb_dim)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, training_data: Data):
        x = torch.tanh(self.lin1(training_data.embedding.weight))
        x = self.lin2(x)
        embedding = nn.Embedding.from_pretrained(x, freeze=True)
        x = self.rgcn1(embedding.weight, training_data.edge_index, training_data.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, training_data.edge_index, training_data.edge_type)
        x = torch.sigmoid(x)
        return x
    
    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)

