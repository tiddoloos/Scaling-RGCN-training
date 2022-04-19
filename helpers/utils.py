from rdflib import Graph
import rdflib
import torch
from collections import Counter

def make_rdf_graph(file_path: str) -> rdflib.Graph:
    format = file_path.split('.')[-1] #work with .nt or n3
    g = Graph()
    data = open(file_path, 'rb')
    g.parse(data, format = format)
    return g

def get_relations(graph, edge):
    relations = set(graph.predicates())
    # makes sure type edge not considered
    relations = list(relations - set(edge))
    return relations

def freq(rel: str, freq_):
        return freq_[rel] if rel in freq_ else 0

def process_rdf_graph(graph_path):
    edge = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    
    graph = make_rdf_graph(graph_path)
    
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
