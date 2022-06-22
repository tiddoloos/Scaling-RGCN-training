from collections import defaultdict
from typing import List, Dict, Tuple


def parse_graph_nt(path: str) -> List[str]:
    with open(path, 'r') as file:
        triples = file.read().replace(' .', '').splitlines()
    return triples

def nodes2type_mapping(graph_triples: List[str]) -> Tuple[List, Dict[str, List]]:
    rel = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
    node2types_dict = defaultdict(list)
    classes = []
    for triple in graph_triples:
        triple_list = triple.split(" ", maxsplit=2)
        if triple_list != ['']:
            s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
            if str(p) == rel.lower() and str(s).split('#')[0] != 'http://swrc.ontoware.org/ontology':
                node2types_dict[s].append(o)
                classes.append(o)
    classes = sorted(list(set(classes)))
    return classes, node2types_dict 

def get_node_mappings_dict(graph_triples: List[str]) -> Tuple[Dict[str, str], Dict[str, List]]:
    sumNode2orgNode_dict = defaultdict(list)
    orgNode2sumNode_dict = defaultdict()
    for triple in graph_triples:
        triple_list = triple.split(" ", maxsplit=2)
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
        sg_labels = [0 for _ in range(num_classes)]
        for node in orgNodes:
            types = org2type_dict[node]
            for t in types:
                sg_labels[labels_dict[t]] += 1
        div = max(1, len(orgNodes))
        sg_labels[:] = [x / div for x in sg_labels]
        sum2type_enc[sumNode] = sg_labels
    return sum2type_enc
