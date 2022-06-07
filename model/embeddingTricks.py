import torch

from torch import nn, Tensor
from typing import List

from graphdata.graph import Graph

def get_tensor_list(graph: Graph, sum_graphs: list, emb_dim: int) -> List[Tensor]:
    tensors = []
    for sum_graph in sum_graphs:
        embedding_tensor = torch.rand(graph.num_nodes, emb_dim, requires_grad=False)
        for orgNode, idx in graph.node_to_enum.items():
            if orgNode in sum_graph.orgNode2sumNode_dict.keys():
                sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
                if sumNode in sum_graph.node_to_enum:
                    embedding_tensor[idx] = sum_graph.training_data.embedding[sum_graph.node_to_enum[sumNode]].detach()
        tensors.append(embedding_tensor)
    return tensors

def stack_embeddings(graph: Graph, sum_graphs: list, emb_dim: int) -> None:
    tensors = get_tensor_list(graph, sum_graphs, emb_dim)
    stacked_emb = torch.stack(tensors)
    graph.training_data.embedding=stacked_emb.detach()

def concat_embeddings(graph: Graph, sum_graphs: list, emb_dim: int) -> None:
    #make concatted tensor of embedding
    tensors = get_tensor_list(graph, sum_graphs, emb_dim)
    concat_emb = torch.concat(tensors, dim=-1)
    graph.training_data.embedding=concat_emb.detach()

#embdding bag
def sum_embeddings(graph: Graph, sum_graphs: List[Graph], emb_dim) -> None:
    # summing of embeddings
    tensors = get_tensor_list(graph, sum_graphs, emb_dim)
    summed_embedding = sum(tensors)
    mean_embedding = summed_embedding / len(sum_graphs)
    graph.training_data.embedding=mean_embedding.detach()

# def sum_embeddings(graph: Graph, sum_graphs: List[Graph], emb_dim) -> None:
#     # summing of the embeddings
#     summed_embedding = torch.rand(graph.num_nodes, emb_dim, requires_grad=False)
#     for orgNode, idx in graph.node_to_enum.items():
#         # makes a vectors with zeros and add embedings for specific idx
#         sum_weight = torch.zeros(1, emb_dim)
#         for sum_graph in sum_graphs:
#             #make sure to only continue if org node is linked to a sumNode
#             if orgNode in sum_graph.orgNode2sumNode_dict:
#                 sumNode = sum_graph.orgNode2sumNode_dict[orgNode]
#                 if sumNode in sum_graph.node_to_enum:
#                     sum_weight += sum_graph.embedding.[sum_graph.node_to_enum[sumNode]]
#         if torch.count_nonzero(sum_weight):
#             summed_embedding[idx] = sum_weight.detach()
#     graph.embedding=nn.Embedding.from_pretrained(summed_embedding, freeze=True)
