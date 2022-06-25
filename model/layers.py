import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data 


class Emb_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, num_nodes: int, emb_dim: int, _) -> None:
        super(Emb_Layers, self).__init__()
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations, num_bases=None)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations, num_bases=None)

        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')
        # this results in worse performance 
        # nn.init.kaiming_uniform_(self.embedding.weight, mode='fan_in')

    def forward(self, training_data: Data) -> Tensor:
        # x = torch.sigmoid(self.embedding.weight)
        x = self.rgcn1(self.embedding.weight, training_data.edge_index, training_data.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, training_data.edge_index, training_data.edge_type)
        x = torch.sigmoid(x)
        return x
    
    def reset_embedding(self, num_nodes: int, emb_dim: int) -> None:
        self.embedding = nn.Embedding(num_nodes, emb_dim)

    def load_embedding(self, embedding: Tensor, freeze: bool=True) -> None:
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)

    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor, grad: bool = True) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.weight.requires_grad = grad
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.bias.requires_grad = grad
        self.rgcn1.root = torch.nn.Parameter(root_1)
        self.rgcn1.root.requires_grad = grad

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.weight.requires_grad = grad
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.bias.requires_grad = grad
        self.rgcn2.root = torch.nn.Parameter(root_2)
        self.rgcn2.root.requires_grad = grad



class Emb_ATT_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, _, emb_dim: int, num_sums: int) -> None:
        super(Emb_ATT_Layers, self).__init__()
        self.embedding = None
        self.att = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_sums, dropout=0.0)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations, num_bases=None)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations, num_bases=None)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, training_data: Data) -> Tensor:
        # x = torch.sigmoid(self.embedding)
        attn_output, att_weights = self.att(self.embedding, self.embedding, self.embedding)
        # print(att_weights.size())
        # print(att_weights)
        # print(att_weights[0])
        # print(att_weights[1])
        # print(att_weights[2])
        # x = torch.sigmoid(attn_output[0])
        x = attn_output[0]
        x = self.rgcn1(x, training_data.edge_index, training_data.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, training_data.edge_index, training_data.edge_type)
        x = torch.softmax(x)
        return x
    
    def load_embedding(self, embedding: Tensor, grad: bool=False) -> None:
        self.embedding = nn.Parameter(embedding, requires_grad=grad)

    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor, grad: bool = True) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.weight.requires_grad = grad
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.bias.requires_grad = grad
        self.rgcn1.root = torch.nn.Parameter(root_1)
        self.rgcn1.root.requires_grad = grad

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.weight.requires_grad = grad
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.bias.requires_grad = grad
        self.rgcn2.root = torch.nn.Parameter(root_2)
        self.rgcn2.root.requires_grad = grad


class Emb_MLP_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, num_nodes: int, emb_dim: int, num_sums: int):
        in_f = num_sums * emb_dim
        out_f = round((in_f/2) * 3 + num_labels)
        super(Emb_MLP_Layers, self).__init__()
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.lin1 = nn.Linear(in_features=in_f, out_features=out_f)
        self.lin2 = nn.Linear(in_features=out_f, out_features=emb_dim)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations, num_bases=None)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations, num_bases=None)
        nn.init.kaiming_uniform_(self.lin1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.lin2.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, training_data: Data):
        # try relu
        x = torch.sigmoid(self.lin1(self.embedding.weight))
        x = torch.sigmoid(self.lin2(x))
        # x = self.lin2(x)
        x = self.rgcn1(x, training_data.edge_index, training_data.edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, training_data.edge_index, training_data.edge_type)
        x = torch.sigmoid(x)
        return x
    
    def load_embedding(self, embedding: Tensor, freeze: bool=True) -> None:
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)

    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor, grad: bool = True) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.weight.requires_grad = grad
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.bias.requires_grad = grad
        self.rgcn1.root = torch.nn.Parameter(root_1)
        self.rgcn1.root.requires_grad = grad

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.weight.requires_grad = grad
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.bias.requires_grad = grad
        self.rgcn2.root = torch.nn.Parameter(root_2)
        self.rgcn2.root.requires_grad = grad
