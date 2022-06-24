import torch

from torch import Tensor
from typing import List

from graphdata.graph import Graph

def get_tensor_list(graph: Graph, sum_graphs: list, emb_dim: int) -> List[Tensor]:
    '''This function loops over each summary graph.
    It creates for each summary graph a new emebdding with the size of the orginal graph embedding (shape: num_nodes, emb_dim).
    For each node in the original graph it checks if the node is mapped to a summary node and if so it it copies the embedding
    of the summary node to the new constructed emdedding (on idx). The resulting tensors are returned in a list.
    Return:
        List[Tensor]
    '''
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
    '''make a stacked (3d) embedding tensor.
    Resulting embedding tensor is of size: (num_sums, num_graph_nodes, emb_dim))
    '''
    tensors = get_tensor_list(graph, sum_graphs, emb_dim)
    stacked_emb = torch.stack(tensors)
    graph.training_data.embedding=stacked_emb.detach()

def concat_embeddings(graph: Graph, sum_graphs: list, emb_dim: int) -> None:
    '''make concatted (2d) tensor of embedding. 
    Resulting embedding tensor is of size: (num_graph_nodes, (num_summaries * emb_dim))
    '''
    tensors = get_tensor_list(graph, sum_graphs, emb_dim)
    concat_emb = torch.concat(tensors, dim=-1)
    graph.training_data.embedding=concat_emb.detach()

def sum_embeddings(graph: Graph, sum_graphs: List[Graph], emb_dim) -> None:
    '''construct a new (2d) embedding tensor.
    Resulting embedding tensor is of size: (num_graph_nodes, emb_dim))
    '''
    tensors = get_tensor_list(graph, sum_graphs, emb_dim)
    summed_embedding = sum(tensors)
    # mean_embedding = summed_embedding / len(sum_graphs)
    graph.training_data.embedding=summed_embedding.detach()