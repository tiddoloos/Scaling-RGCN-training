from copy import deepcopy
from collections import defaultdict
from rdflib import Graph as rdfGraph
from torch import nn
from torch_geometric.data import Data
from typing import List, Dict, Tuple

from graphdata.graphUtils import process_rdf_graph
from graphdata.graph import Graph


def nodes2type_mapping(graph: rdfGraph) -> Tuple[List, Dict[str, List]]:
    rel = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    node2types_dict = defaultdict(list)
    classes = []
    for s, p, o in graph:
        if str(p).lower() == rel.lower() and str(s).split('#')[0] != 'http://swrc.ontoware.org/ontology':
            s_ = str(s).lower()
            type_ = str(o).lower()
            node2types_dict[s_].append(type_)
            classes.append(type_)
    classes = sorted(list(set(classes)))
    return classes, node2types_dict 

def get_node_mappings_dict(graph: rdfGraph) -> Tuple[Dict[str, str], Dict[str, List]]:
    sumNode2orgNode_dict = defaultdict(list)
    orgNode2sumNode_dict = defaultdict()
    for s, _, o in graph:
        s_ = str(s).lower() 
        o_ = str(o).lower()
        sumNode2orgNode_dict[s_].append(o_)
        orgNode2sumNode_dict[o_] = s_
    sumNode2orgNode_dict = dict(sorted(sumNode2orgNode_dict.items()))
    orgNode2sumNode_dict = dict(sorted(orgNode2sumNode_dict.items()))
    return orgNode2sumNode_dict, sumNode2orgNode_dict

def encode_node_labels(sumNode2orgNode_dict: defaultdict(list), org2type_dict: defaultdict(list), labels_dict: dict, num_classes: int) -> Tuple[Dict[str, List], Dict[str, List]]:
    sum2type_vec = defaultdict()
    org2type_vec = defaultdict()
    for sumNode, orgNodes in sumNode2orgNode_dict.items():
        sg_labels = [0 for _ in range(num_classes)]
        for node in orgNodes:
            g_labels = [0 for _ in range(num_classes)]
            types = org2type_dict[node]
            for t in types:
                sg_labels[labels_dict[t]] += 1
                g_labels[labels_dict[t]] += 1
            org2type_vec[node] = g_labels
        div = 1
        if len(orgNodes) > 0:
            div = len(orgNodes)
        sg_labels[:] = [x / div for x in sg_labels]
        sum2type_vec[sumNode] = sg_labels
    return sum2type_vec, org2type_vec 

def get_graph_data(org_graph: rdfGraph, sum_graph: rdfGraph, map_graph: rdfGraph, emb_dim=None, org_only=False) ->  Tuple[int, int, Graph]:
    if org_only:
        edge_index, edge_type, node_to_enum, num_nodes, sorted_nodes, relations_dict  = process_rdf_graph(org_graph)
        oGraph = Graph(node_to_enum, num_nodes, sorted_nodes, relations_dict, None, None, None, None, None, None)
        oGraph.training_data = Data(edge_index = edge_index)
        oGraph.training_data.edge_type = edge_type
        oGraph.training_data.embedding = nn.Embedding(num_nodes, emb_dim)
        return oGraph

    else:
        edge_index, edge_type, node_to_enum, num_nodes, sorted_nodes, relations_dict = process_rdf_graph(sum_graph)
        classes, org2type_dict = nodes2type_mapping(org_graph)
        orgNode2sumNode_dict, sumNode2orgNode_dict = get_node_mappings_dict(map_graph)
        enum_classes = {lab: i for i, lab in enumerate(classes)}
        sum2type_enc, org2type_enc  = encode_node_labels(sumNode2orgNode_dict, deepcopy(org2type_dict), enum_classes, len(classes))
        sGraph = Graph(node_to_enum, num_nodes, sorted_nodes, relations_dict, orgNode2sumNode_dict, sumNode2orgNode_dict, org2type_dict, org2type_enc, sum2type_enc, None)
        sGraph.training_data = Data(edge_index = edge_index)
        sGraph.training_data.edge_type = edge_type
        sGraph.training_data.embedding = nn.Embedding(num_nodes, emb_dim)
        return enum_classes, len(classes), sGraph
