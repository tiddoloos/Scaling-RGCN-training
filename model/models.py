import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch_geometric.nn import RGCNConv

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
class emb_sum_layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, emb_dim) -> None:
        self.emb_dim = emb_dim
        super(emb_sum_layers, self).__init__()
        self.embedding = None
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def sum_embeddings(self, graph, sum_graphs: list) -> None:
        #summing of the embeddings
        summed_embedding = torch.rand(graph.num_nodes, self.emb_dim, requires_grad=False)
        for orgNode, idx in graph.node_to_enum.items():
            sum_weight = torch.zeros(1, self.emb_dim)
            for sum_graph in sum_graphs:
                #make sure to only continue if org node is linked to a sumNode
                if orgNode in sum_graph.orgNode2sumNode_dict:
                    sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
                    if sumNode in sum_graph.node_to_enum:
                        sum_weight += sum_graph.embedding.weight[sum_graph.node_to_enum[sumNode]]
            if torch.count_nonzero(sum_weight):
                summed_embedding[idx] = sum_weight.detach()
        self.embedding=nn.Embedding.from_pretrained(summed_embedding, freeze=False)

    def init_embeddings(self, num_nodes:int) -> None:
        self.embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=self.emb_dim)

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

class emb_mlp_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, in_f: int, out_f: int, emb_dim: int):
        self.emb_dim = emb_dim
        self.concat_emb = None
        super(emb_mlp_Layers, self).__init__()
        self.lin1 = nn.Linear(in_features=in_f, out_features=out_f)
        self.lin2 = nn.Linear(in_features=out_f, out_features=emb_dim)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def forward(self, edge_index: Tensor, edge_type: Tensor):
        x = torch.tanh(self.lin1(self.concat_emb.weight))
        x = self.lin2(x)
        embedding = nn.Embedding.from_pretrained(x, freeze=True)
        x = self.rgcn1(embedding.weight, edge_index, edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = torch.sigmoid(x)
        return x
    
    def concat_embeddings(self, graph, sum_graphs: list):
        #make concatted tensor of embeddings
        tensors = []
        for sum_graph in sum_graphs:
            embedding_tensor = torch.rand(graph.num_nodes, self.emb_dim, requires_grad=False)
            for orgNode, idx in graph.node_to_enum.items():
                if orgNode in sum_graph.orgNode2sumNode_dict.keys():
                    sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
                    if sumNode in sum_graph.node_to_enum:
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

class emb_att_Layers(nn.Module):
    def __init__(self, num_relations: int, hidden_l: int, num_labels: int, num_sums: int, emb_dim):
        self.emb_dim = emb_dim
        super(emb_att_Layers, self).__init__()
        self.embedding = None
        self.att = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_sums)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_l, num_labels, num_relations)
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_in')

    def init_embeddings(self, num_nodes:int):
        self.embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=self.emb_dim)

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        attn_output, attn_output_weights = self.att(self.embedding, self.embedding, self.embedding)
        x = self.rgcn1(attn_output[0], edge_index, edge_type)
        x = F.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = torch.sigmoid(x)
        return x
    
    def stack_embeddings(self, graph, sum_graphs: list):
        #make stacked tensor of embedding
        tensors = []
        for sum_graph in sum_graphs:
            embedding_tensor = torch.rand(graph.num_nodes, self.emb_dim, requires_grad=False)
            for orgNode, idx in graph.node_to_enum.items():
                if orgNode in sum_graph.orgNode2sumNode_dict.keys():
                    sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
                    if sumNode in sum_graph.node_to_enum:
                        embedding_tensor[idx]  = sum_graph.embedding.weight[sum_graph.node_to_enum[sumNode]].detach()
            tensors.append(embedding_tensor)
        stacked_emb = torch.stack(tensors)
        self.embedding=stacked_emb

    def override_params(self, weight_1: Tensor, bias_1: Tensor, root_1: Tensor, weight_2: Tensor, bias_2: Tensor, root_2: Tensor) -> None:
        self.rgcn1.weight = torch.nn.Parameter(weight_1)
        self.rgcn1.bias = torch.nn.Parameter(bias_1)
        self.rgcn1.root = torch.nn.Parameter(root_1)

        self.rgcn2.weight = torch.nn.Parameter(weight_2)
        self.rgcn2.bias = torch.nn.Parameter(bias_2)
        self.rgcn2.root = torch.nn.Parameter(root_2)
