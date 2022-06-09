import rdflib

from collections import defaultdict
from rdflib import Graph as rdfGraph
from typing import List, Dict, Tuple


def make_rdf_graph(file_path: str) -> rdflib.Graph:
    format = file_path.split('.')[-1]
    g = rdfGraph()
    with open(file_path, 'rb') as data:
        g.parse(data, format = format)  
    return g

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


def encode_org_node_labels(org2type_dict: defaultdict(list), labels_dict: dict, num_classes: int) -> Dict[str, List[float]]:
    org2type_enc = defaultdict()
    for node in org2type_dict.keys():
        g_labels = [0 for _ in range(num_classes)]
        types = org2type_dict[node]
        for t in types:
            g_labels[labels_dict[t]] += 1
        org2type_enc[node] = g_labels
    return org2type_enc 

def encode_sum_node_labels(sumNode2orgNode_dict: defaultdict(list), org2type_dict: defaultdict(list), labels_dict: dict, num_classes: int) -> Tuple[Dict[str, List]]:
    sum2type_enc = defaultdict(list)
    for sumNode, orgNodes in sumNode2orgNode_dict.items():
        sg_labels = [0 for _ in range(num_classes)]
        for node in orgNodes:
            types = org2type_dict[node]
            for t in types:
                sg_labels[labels_dict[t]] += 1
        div = 1
        if len(orgNodes) > 0:
            div = len(orgNodes) 
        sg_labels[:] = [x / div for x in sg_labels]
        sum2type_enc[sumNode] = sg_labels
    return sum2type_enc
