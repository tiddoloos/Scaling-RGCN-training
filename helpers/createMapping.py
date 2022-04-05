from collections import defaultdict
from helpers.utils import make_rdf_graph

def nodes2type_mapping(path):
    rel = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    graph = make_rdf_graph(path)
  
    node2types_dict = defaultdict(list)
    labels = []
    for s, p, o in graph:
        if str(p).lower() == rel.lower() and str(s).split('#')[0] != 'http://swrc.ontoware.org/ontology':
            s_ = str(s).lower()
            type_ = str(o).lower()
            node2types_dict[s_].append(type_)
            labels.append(type_)
    labels = sorted(list(set(labels)))
    return labels, node2types_dict 

def get_node_mappings_dict(path):
    graph = make_rdf_graph(path)
    sumNode2orgNode_dict = defaultdict(list)
    orgNode2sumNode_dict = defaultdict()
    for s, _, o in graph:
        s_ = str(s).lower()
        o_ = str(o).lower()
        sumNode2orgNode_dict[s_].append(o_)
        orgNode2sumNode_dict[o_] = s_
    return orgNode2sumNode_dict, sumNode2orgNode_dict

def vectorize_label_mapping(sumNode2orgNode_dict, org2type_dict, labels_dict, num_classes):
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
        for i in range(len(sg_labels)):
            sg_labels[i] = sg_labels[i] / len(orgNodes)
        sum2type_vec[sumNode] = sg_labels
    return sum2type_vec, org2type_vec 

def main_createMappings(dataset_name):
    org_file = f'data/{dataset_name}/{dataset_name}_complete.n3'
    sum_map_file = f'data/{dataset_name}/{dataset_name}_attr_map.n3'

    orgNode2sumNode_dict, sumNode2orgNode_dict = get_node_mappings_dict(sum_map_file)

    labels, org2type_dict = nodes2type_mapping(org_file)
    labels_dict = {lab: i for i, lab in enumerate(labels)}

    sum2type_vec, org2type_vec  = vectorize_label_mapping(sumNode2orgNode_dict, org2type_dict, labels_dict, len(labels))

    return sum2type_vec, org2type_vec, labels_dict, len(labels), orgNode2sumNode_dict, sumNode2orgNode_dict 
