from data.graphData import Graph
import torch
from torch import device, nn, Tensor
from typing import List

def get_tensor_list(graph: Graph, sum_graphs: list, emb_dim: int)-> List[Tensor]:
    tensors = []
    for sum_graph in sum_graphs:
        embedding_tensor = torch.rand(graph.num_nodes, emb_dim, requires_grad=False)
        for orgNode, idx in graph.node_to_enum.items():
            if orgNode in sum_graph.orgNode2sumNode_dict.keys():
                sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
                if sumNode in sum_graph.node_to_enum:
                    embedding_tensor[idx] = sum_graph.embedding.weight[sum_graph.node_to_enum[sumNode]].detach()
        tensors.append(embedding_tensor)
    return tensors

def stack_embeddings(graph: Graph, sum_graphs: list, emb_dim: int) -> None:
    tensors = get_tensor_list(graph, sum_graphs, emb_dim)
    stacked_emb = torch.stack(tensors)
    # nn.Embeddeing can only handle 2d dimension so keep emb as stacked 3d tensor
    graph.embedding=stacked_emb

def concat_embeddings(graph: Graph, sum_graphs: list, emb_dim: int) -> None:
    #make concatted tensor of embeddings
    #or use stack if dims get to high -> sqeenze in MLP
    tensors = get_tensor_list(graph, sum_graphs, emb_dim)
    concat_emb = torch.concat(tensors, dim=-1)
    graph.embedding=nn.Embedding.from_pretrained(concat_emb, freeze=True)

def sum_embeddings(graph: Graph, sum_graphs: List[Graph], emb_dim) -> None:
        # summing of embeddings
        tensors = get_tensor_list(graph, sum_graphs, emb_dim)
        summed_embedding = sum(tensors) 
        graph.embedding=nn.Embedding.from_pretrained(summed_embedding, freeze=True).to(device)

# def sum_embeddings(graph: Graph, sum_graphs: List[Graph], emb_dim) -> None:
#         # summing of the embeddings
#         summed_embedding = torch.rand(graph.num_nodes, emb_dim, requires_grad=False)
#         for orgNode, idx in graph.node_to_enum.items():
#             sum_weight = torch.zeros(1, emb_dim)
#             for sum_graph in sum_graphs:
#                 #make sure to only continue if org node is linked to a sumNode
#                 if orgNode in sum_graph.orgNode2sumNode_dict:
#                     sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
#                     if sumNode in sum_graph.node_to_enum:
#                         sum_weight += sum_graph.embedding.weight[sum_graph.node_to_enum[sumNode]]
#             if torch.count_nonzero(sum_weight):
#                 summed_embedding[idx] = sum_weight.detach()
#         graph.embedding=nn.Embedding.from_pretrained(summed_embedding, freeze=False)
