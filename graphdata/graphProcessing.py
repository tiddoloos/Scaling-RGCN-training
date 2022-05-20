import rdflib
import torch

from collections import Counter
from rdflib import Graph as rdfGraph
from rdflib.term import URIRef


from collections import defaultdict
from rdflib import Graph as rdfGraph
from typing import List, Dict, Tuple


def make_rdf_graph(file_path: str) -> rdflib.Graph:
    format = file_path.split('.')[-1]
    g = rdfGraph()
    with open(file_path, 'rb') as data:
        g.parse(data, format = format)  
    return g

def get_relations(graph: rdfGraph, edge: URIRef):
    relations = list(set(graph.predicates()))
    # remove type edge
    relations.remove(edge)
    return relations

def freq(rel: str, freq_: Counter):
        return freq_[rel] if rel in freq_ else 0

def process_rdf_graph(graph: rdfGraph):
    edge = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
    rels = get_relations(graph, edge)
    freq_ = Counter(rels)
    relations = sorted(set(rels), key=lambda rel: -freq(rel, freq_))

    subjects = set(graph.subjects())
    objects = set(graph.objects())
    nodes = list(subjects.union(objects))
    sorted_nodes = sorted(nodes)

    # relation to integer idx
    relations_dict = {str(rel).lower(): i for i, rel in enumerate(list(relations))}

    # node to integer idx
    nodes_dict = {str(node).lower(): i for i, node in enumerate(sorted_nodes)}
  
    edge_list = []
    count = 0
    for s, p, o in graph:
        count += 1
        s_ = str(s).lower()
        o_ = str(o).lower()
        p_ = str(p).lower()
        if nodes_dict.get(s_) is not None and  relations_dict.get(p_) is not None and nodes_dict.get(o_) is not None:
            src, dst, rel = nodes_dict[s_], nodes_dict[o_], relations_dict[p_]
            # undirected
            edge_list.append([src, dst, 2 * rel])
            edge_list.append([dst, src, 2 * rel + 1])
    edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))
    edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index, edge_type = edge[:2], edge[2]

    return edge_index, edge_type, nodes_dict, len(sorted_nodes), sorted_nodes, relations_dict

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