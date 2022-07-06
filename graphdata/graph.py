import torch

from typing import List, Dict
from torch_geometric.data import Data
from torch import Tensor


class Graph:
    def __init__(self, name: str, org2type_dict: Dict[str, List[str]]) -> None:
        self.name = name
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

    def init_graph(self, graph_triples: List[str]):
        subjects = set()
        predicates = set()
        objects = set()
        for triple in graph_triples:
            triple_list = triple[:-2].split(" ", maxsplit=2)
            if triple_list != ['']:
                s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
                subjects.add(s)
                predicates.add(p)
                objects.add(o)

        # remove type edges from predicates
        edges = ['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', '<type>']
        for edge in edges:
            if edge in predicates:
                predicates.remove(edge)

        nodes = list(subjects.union(objects))
        self.nodes = sorted(nodes)
        self.num_nodes = len(nodes)

        # relation to integer idx
        self.relations = {str(rel): i for i, rel in enumerate(list(predicates))}
        # node to integer idx
        self.node_to_enum = {str(node): i for i, node in enumerate(self.nodes)}
    
        edge_list = []
        for triple in graph_triples:
            triple_list = triple[:-2].split(" ", maxsplit=2)
            if triple_list != ['']:
                s_, p_, o_ = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
                if self.node_to_enum.get(s_) is not None and  self.relations.get(p_) is not None and self.node_to_enum.get(o_) is not None:
                    src, dst, rel = self.node_to_enum[s_], self.node_to_enum[o_], self.relations[p_]
                    edge_list.append([src, dst, 2 * rel])
                    edge_list.append([dst, src, 2 * rel + 1])
        edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))
        edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index, edge_type = edge[:2], edge[2]
        
        self.training_data = Data(edge_index=edge_index)
        self.training_data.edge_type = edge_type
