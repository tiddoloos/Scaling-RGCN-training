import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class transfer_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int) -> None:
        super(transfer_Layers, self).__init__()
        self.embedding = None
        self.rgcn1 = RGCNConv(in_channels=64, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def attention():
        pass

    def init_embeddings(self, num_nodes:int):
        self.embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=64)

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        x = self.rgcn1(self.embedding.weight, edge_index, edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = torch.sigmoid(x)
        return x
    
    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)
        