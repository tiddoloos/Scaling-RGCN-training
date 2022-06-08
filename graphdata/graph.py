import torch

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Tuple
from torch_geometric.data import Data
from torch import Tensor
from rdflib import Graph as rdfGraph
from sklearn.model_selection import train_test_split

from helpers import timing
from graphdata.graphProcessing import process_rdf_graph, get_node_mappings_dict, encode_node_labels


@dataclass
class Graph:
    node_to_enum: Dict[str, int]
    num_nodes: int
    nodes: List[str]
    relations: Dict[str, int]
    orgNode2sumNode_dict: Dict[str, List[str]]
    sumNode2orgNode_dict: Dict[str, List[str]]
    org2type_dict: Dict[str, List[str]]
    org2type: Dict[str, List[str]]
    sum2type: Dict[str, List[str]]
    training_data: Data
    embedding: Tensor


def get_graph_data(org_graph: rdfGraph, sum_graph: rdfGraph, map_graph: rdfGraph, org2type_dict: Dict[str, str],
                    enum_classes: Dict[str, int], num_classes: int, org=False) ->  Graph:

    if org:
        edge_index, edge_type, node_to_enum, num_nodes, sorted_nodes, relations_dict  = process_rdf_graph(org_graph)
        oGraph = Graph(node_to_enum, num_nodes, sorted_nodes, relations_dict, None, None, None, None, None, None, None)
        oGraph.training_data = Data(edge_index = edge_index)
        oGraph.training_data.edge_type = edge_type
        return oGraph

    else:
        edge_index, edge_type, node_to_enum, num_nodes, sorted_nodes, relations_dict = process_rdf_graph(sum_graph)
        orgNode2sumNode_dict, sumNode2orgNode_dict = get_node_mappings_dict(map_graph)
        sum2type_enc, org2type_enc  = encode_node_labels(sumNode2orgNode_dict, org2type_dict, enum_classes, num_classes)
        sGraph = Graph(node_to_enum, num_nodes, sorted_nodes, relations_dict, orgNode2sumNode_dict, sumNode2orgNode_dict, org2type_dict, org2type_enc, sum2type_enc, None, None)
        sGraph.training_data = Data(edge_index = edge_index)
        sGraph.training_data.edge_type = edge_type
        return sGraph

def remove_eval_data(X_test: List[int], orgGraph: Graph, sumGraph: Graph, enum_classes: Dict[str, int], num_classes: int) -> None:
    """This funtion updates the sum2type dict by removing the test data.
    Avoids test set leakage because orignal node maps to a summary nodes which maps to a type (predicted class).
    """ 
    # make a copy to preserve orginal data in the data opject
    org2type = deepcopy(sumGraph.org2type_dict)
    for orgNode, value in orgGraph.node_to_enum.items():
        if value in X_test:
            org2type[orgNode].clear()
    sumGraph.sum2type, _  =  encode_node_labels(sumGraph.sumNode2orgNode_dict, org2type, enum_classes, num_classes)

def get_idx_labels(graph: Graph, node2type: Dict[str, List[float]]) -> Tuple[List[int], List[int]]:
    train_indices = list()
    train_labels = list()
    for node, labs in node2type.items():
        if sum(list(labs)) != 0.0 and graph.node_to_enum.get(node) is not None:
            train_indices.append(graph.node_to_enum[node])
            train_labels.append(list(labs))
    return train_indices, train_labels

def make_graph_trainig_data(orgGraph: Graph, sumGraphs: List[Graph], enum_classes: Dict[str, int], num_classes: int):
    g_idx, g_labels = get_idx_labels(orgGraph, sumGraphs[0].org2type)
    X_train, X_test, y_train, y_test = train_test_split(g_idx, g_labels,  test_size=0.2, random_state=1) 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    orgGraph.training_data.x_train = torch.tensor(X_train, dtype = torch.long)
    orgGraph.training_data.x_test = torch.tensor(X_test) 
    orgGraph.training_data.x_val = torch.tensor(X_val, dtype = torch.long)
    orgGraph.training_data.y_val = torch.tensor(y_val, dtype = torch.long)
    orgGraph.training_data.y_train = torch.tensor(y_train, dtype = torch.long)
    orgGraph.training_data.y_test = torch.tensor(y_test)

    print("ORIGINAL GRAPH STATISTICS")
    print(f"num Nodes = {orgGraph.num_nodes}")
    print(f"num Relations = {len(orgGraph.relations.keys())}")
    print(f"num Classes = {num_classes}")
    timing.log('ORGINAL GRPAH LOADED')
    to_remove = X_test + X_val

    for sumGraph in sumGraphs:
        remove_eval_data(to_remove, orgGraph, sumGraph, enum_classes, num_classes)

        sg_idx, sg_labels = get_idx_labels(sumGraph, sumGraph.sum2type)
        sumGraph.training_data.x_train = torch.tensor(sg_idx, dtype = torch.long)
        sumGraph.training_data.y_train = torch.tensor(sg_labels)
        
        print("SUMMARY GRAPH STATISTICS")
        print(f"num Nodes = {sumGraph.num_nodes}")
        print(f"num Relations= {len(sumGraph.relations.keys())}")

        # Assertion error: if more relations in summary graph than in original graph
        assert len(sumGraph.relations.keys()) ==  len(orgGraph.relations.keys()), 'number of relations in summary graph and original graph differ'