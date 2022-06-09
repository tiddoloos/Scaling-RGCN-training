import torch

from collections import Counter
from dataclasses import dataclass
from typing import List, Dict
from torch_geometric.data import Data
from torch import Tensor
from rdflib import Graph as rdfGraph
from rdflib.term import URIRef



class Graph:
    def __init__(self, org2type_dict) -> None:
        self.nodes: List[str] = None
        self.node_to_enum: Dict[str, int] = None
        self.num_nodes: int = None
        self.relations: Dict[str, int] = None
        self.orgNode2sumNode_dict: Dict[str, List[str]] = None
        self.sumNode2orgNode_dict: Dict[str, List[str]] = None
        self.org2type_dict: Dict[str, List[str]] = org2type_dict
        self.org2type: Dict[str, List[str]] = None
        self.sum2type: Dict[str, List[str]] = None
        self.training_data: Data = None
        self.embedding: Tensor = None
    
    def get_relations(self, graph: rdfGraph, edge: URIRef):
        # remove type edge
        relations = list(set(graph.predicates()))
        relations.remove(edge)
        return relations

    def freq(self, rel: str, freq_: Counter):
            return freq_[rel] if rel in freq_ else 0

    def init_graph(self, graph: rdfGraph):
        edge = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
        rels = self.get_relations(graph, edge)
        freq_ = Counter(rels)
        relations = sorted(set(rels), key=lambda rel: - self.freq(rel, freq_))

        subjects = set(graph.subjects())
        objects = set(graph.objects())
        nodes = list(subjects.union(objects))
        self.nodes = sorted(nodes)
        self.num_nodes = len(nodes)

        # relation to integer idx
        self.relations = {str(rel).lower(): i for i, rel in enumerate(list(relations))}
        # node to integer idx
        self.node_to_enum = {str(node).lower(): i for i, node in enumerate(self.nodes)}
    
        edge_list = []
        count = 0
        for s, p, o in graph:
            count += 1
            s_ = str(s).lower()
            o_ = str(o).lower()
            p_ = str(p).lower()
            if self.node_to_enum.get(s_) is not None and  self.relations.get(p_) is not None and self.node_to_enum.get(o_) is not None:
                src, dst, rel = self.node_to_enum[s_], self.node_to_enum[o_], self.relations[p_]
                # undirected
                edge_list.append([src, dst, 2 * rel])
                edge_list.append([dst, src, 2 * rel + 1])
        edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))
        edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index, edge_type = edge[:2], edge[2]
        self.training_data = Data(edge_index=edge_index)
        self.training_data.edge_type = edge_type
