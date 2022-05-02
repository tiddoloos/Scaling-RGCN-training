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

    def sum_embeddings(self, graph, sum_graphs: list):
        #summing of the embeddings
        summed_embedding = torch.rand(graph.num_nodes, 64, requires_grad=False)
        for orgNode, idx in graph.node_to_enum.items():
            sum_weight = torch.zeros(1, 64)
            for sum_graph in sum_graphs:
                #make sure to only continue if org node is linked to a sumNode
                if orgNode in sum_graph.orgNode2sumNode_dict.keys():
                    sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
                    sum_weight += sum_graph.embedding.weight[sum_graph.node_to_enum[sumNode]]
            if torch.count_nonzero(sum_weight):
                summed_embedding[idx] = sum_weight.detach()
        self.embedding=nn.Embedding.from_pretrained(summed_embedding, freeze=False)

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
    
##############################################################################################################


class baseline_Layers(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_l: int, num_labels: int) -> None:
        super(baseline_Layers, self).__init__()
        self.rgcn1 = RGCNConv(in_channels=num_nodes, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        x = self.rgcn1(None, edge_index, edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = torch.sigmoid(x)
        return x


###############################################################################################################

class mlp_RGCN_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int):
        self.concat_emb = None
        super(mlp_RGCN_Layers, self).__init__()
        self.lin1 = nn.Linear(in_features=128, out_features=32)
        self.lin2 = nn.Linear(in_features=32, out_features=64)
        self.rgcn1 = RGCNConv(in_channels=64, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, edge_index: Tensor, edge_type: Tensor):
        x = F.tanh(self.lin1(self.concat_emb.weight))
        x = self.lin2(x)
        embedding = nn.Embedding.from_pretrained(x, freeze=True)
        x = self.rgcn1(embedding.weight, edge_index, edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = torch.sigmoid(x)
        return x
    
    def concat_embeddings(self, graph, sum_graphs: list):
        #make stacked tensor of emneddings
        tensors = []
        for sum_graph in sum_graphs:
            embedding_tensor = torch.rand(graph.num_nodes, 64, requires_grad=False)
            #summing of the embeddings
            for orgNode, idx in graph.node_to_enum.items():
                if orgNode in sum_graph.orgNode2sumNode_dict.keys():
                    sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
                    embedding_tensor[idx]  = sum_graph.embedding.weight[sum_graph.node_to_enum[sumNode]].detach()
            tensors.append(embedding_tensor)

        #or use stack if dims get to high -> sqeenze in MLP
        concat_emb = torch.concat(tensors, dim=-1)
        self.concat_emb=nn.Embedding.from_pretrained(concat_emb, freeze=True)
    
    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)

######################################################################################################################

class attention_Layers(nn.Module):
    def __init__(self, num_sums, num_relations: int, hidden_l: int, num_labels: int):
        super(attention_Layers, self).__init__()
        self.embedding = None
        self.att = nn.MultiheadAttention(embed_dim=64, num_heads=num_sums)
        self.rgcn1 = RGCNConv(in_channels=64, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')


    def init_embeddings(self, num_nodes:int):
        self.embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=64)

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        # attn_output, attn_output_weights = att()
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
