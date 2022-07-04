from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple
from graphdata.graph import Graph


def parse_graph_nt(path: str) -> List[str]:
    with open(path, 'r') as file:
        triples = file.read().splitlines()
    return triples

def get_classes(graph_triples: List[str]):
    rel = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
    class_count: dict = defaultdict(int)
    for triple in graph_triples:
        triple_list = triple[:-2].split(" ", maxsplit=2)
        if triple_list != ['']:
            s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
            if str(p) == rel.lower() and str(s).split('#')[0] != 'http://swrc.ontoware.org/ontology':
               class_count[str(o)] += 1
    # print(class_count)
    # adjust threshold to exclude less occuring classes than threshold
    threshold = 100
    c_d = dict((k, v) for k, v in class_count.items() if v >= threshold)
    return sorted(list(c_d.keys()))

def nodes2type_mapping(graph_triples: List[str], classes: List[str]) -> Tuple[List, Dict[str, List]]:
    rel = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
    node2types_dict = defaultdict(set)
    for triple in graph_triples:
        triple_list = triple[:-2].split(" ", maxsplit=2)
        if triple_list != ['']:
            s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
            if str(p) == rel.lower() and str(s).split('#')[0] != 'http://swrc.ontoware.org/ontology' and str(o) in classes:
                node2types_dict[s].add(o)
    multi_label = 0
    one_label = 0
    for nodes, type in node2types_dict.items():
        if len(type) >=2:
            # print(type)
            multi_label += 1
        else:
            one_label += 1
    # print(multi_label)
    # print(one_label)
    return node2types_dict 

def get_node_mappings_dict(graph_triples: List[str]) -> Tuple[Dict[str, str], Dict[str, List]]:
    sumNode2orgNode_dict = defaultdict(list)
    orgNode2sumNode_dict = defaultdict()
    for triple in graph_triples:
        triple_list = triple[:-2].split(" ", maxsplit=2)
        if triple_list != ['']:
            s, _, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
            sumNode2orgNode_dict[s].append(o)
            orgNode2sumNode_dict[o] = s
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

def encode_sum_node_labels(sumNode2orgNode_dict: defaultdict(list), org2type_dict: defaultdict(list), labels_dict: dict, num_classes: int) -> Dict[str, List]:
    sum2type_enc = defaultdict(list)
    for sumNode, orgNodes in sumNode2orgNode_dict.items():
        sg_labels = [0.0 for _ in range(num_classes)]
        for node in orgNodes:
            types = org2type_dict[node]
            for t in types:
                sg_labels[labels_dict[t]] += 1.0
        div = max(1, len(orgNodes))
        sg_labels[:] = [x / div for x in sg_labels]
        sum2type_enc[sumNode] = sg_labels
    return sum2type_enc

def remove_eval_data(X_eval: List[int], orgGraph):
        org2type_pruned = deepcopy(orgGraph.org2type_dict)
        X_eval_set = set(X_eval)
        for orgNode, idx in orgGraph.node_to_enum.items():
            if idx in X_eval_set:
                org2type_pruned[orgNode].clear()
        return org2type_pruned

def get_idx_labels(graph: Graph, node2type) -> Tuple[List[int], List[int]]:
    train_indices: list = []
    train_labels: list = []
    for node, labs in node2type.items():
        if sum(list(labs)) != 0.0 and graph.node_to_enum.get(node) is not None:
            train_indices.append(graph.node_to_enum[node])
            train_labels.append(list(labs))
    return train_indices, train_labels