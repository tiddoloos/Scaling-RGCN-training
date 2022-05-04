from copy import copy
from collections import defaultdict
from typing import Tuple, Dict, List
from helpers.utils import make_rdf_graph

def nodes2type_mapping(path: str) -> Tuple[List, Dict[str, List]]:
    rel = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    node2types_dict = defaultdict(list)
    graph = make_rdf_graph(path)
    classes = []
    for s, p, o in graph:
        if str(p).lower() == rel.lower() and str(s).split('#')[0] != 'http://swrc.ontoware.org/ontology':
            s_ = str(s).lower()
            type_ = str(o).lower()
            node2types_dict[s_].append(type_)
            classes.append(type_)
    classes = sorted(list(set(classes)))
    return classes, node2types_dict 

def get_node_mappings_dict(path: str) -> Tuple[Dict[str, str], Dict[str, List]]:
    graph = make_rdf_graph(path)
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

def encode_label_mapping(sumNode2orgNode_dict: defaultdict(list), org2type_dict: defaultdict(list), labels_dict: dict, num_classes: int) -> Tuple[Dict[str, List], Dict[str, List]]:
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

def main_createMappings(org_file: str, map_file: str) -> Tuple[Dict, Dict, Dict, int, Dict, Dict, Dict, Dict]:
    classes, org2type_dict = nodes2type_mapping(org_file)
    orgNode2sumNode_dict, sumNode2orgNode_dict = get_node_mappings_dict(map_file)
    enum_classes = {lab: i for i, lab in enumerate(classes)}

    #commented: statements to check the change in the dict -> 
    # cop = copy(org2type_dict)
    sum2type_enc, org2type_enc  = encode_label_mapping(sumNode2orgNode_dict, copy(org2type_dict), enum_classes, len(classes))
    # if cop == org2type_dict:
    #     print('TRUE')
    # print(org2type_dict)
    return sum2type_enc, org2type_enc, enum_classes, len(classes), orgNode2sumNode_dict, sumNode2orgNode_dict, org2type_dict
