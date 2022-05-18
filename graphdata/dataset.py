import torch
from torch import nn

from typing import Tuple, List, Dict
from os import listdir
from os.path import isfile, join
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from graphdata.graphUtils import make_rdf_graph, process_rdf_graph
from graphdata.graphConfig import get_graph_data, encode_node_labels
from graphdata.graph import Graph

class Dataset:
    def __init__(self, name: str) -> None:
        self.org_path = f'./graphdata/{name}/{name}_complete.nt'
        self.sum_path = f'./graphdata/{name}/attr/sum/'
        self.map_path = f'./graphdata/{name}/attr/map/'
        self.sumGraphs = []
        self.orgGraph = None
        self.enum_classes = None
        self.num_classes = None
    
    def remove_test_data(self, X_test: List[int]) -> None:
        """This funtion updates the sum2type dict by removing the test data.
        Avoids test set leakage because orignal node maps to a summary nodes which maps to a type (predicted class).
        """ 
        for sumGraph in self.sumGraphs:
            # make a copy to preserve orginal data in the data opject
            org2type = sumGraph.org2type_dict
            for orgNode, value in self.orgGraph.node_to_enum.items():
                if value in X_test:
                    org2type[orgNode].clear()
            sumGraph.sum2type, _  =  encode_node_labels(sumGraph.sumNode2orgNode_dict, org2type, self.enum_classes, self.num_classes)

    def get_idx_labels(self, graph: Graph, node2type: Dict[str, List[float]]) -> Tuple[List[int], List[int]]:
        train_indices = list()
        train_labels = list()
        for node, labs in node2type.items():
            if sum(list(labs)) != 0.0 and graph.node_to_enum.get(node) is not None:
                train_indices.append(graph.node_to_enum[node])
                train_labels.append(list(labs))
        return train_indices, train_labels

    def get_file_names(self) -> Tuple[List[str], List[str]]:
        sum_files = [f for f in listdir(self.sum_path) if not f.startswith('.') if isfile(join(self.sum_path, f))]
        map_files = [f for f in listdir(self.map_path) if not f.startswith('.') if isfile(join(self.map_path, f))]
        assert len(sum_files) == len(map_files), f'for every summary file there needs to be a map file.{sum_files} and {map_files}'
        return sorted(sum_files), sorted(map_files)

    def init_dataset(self, emb_dim: int) -> None:
        rdf_org_graph = make_rdf_graph(self.org_path)
        sum_files, map_files = self.get_file_names()

        # init summary graph data
        for i in range(len(sum_files)):
            sum_path = f'{self.sum_path}/{sum_files[i]}'
            map_path = f'{self.map_path}/{map_files[i]}'
            rdf_sum_graph = make_rdf_graph(sum_path)
            rdf_map_graph = make_rdf_graph(map_path)
            self.enum_classes, self.num_classes, sGraph = get_graph_data(rdf_org_graph, rdf_sum_graph, rdf_map_graph, emb_dim=emb_dim)
            self.sumGraphs.append(sGraph)

        # self.enum_classes, self.num_classes = enum_classes, num_classes
        # init original graph data
        self.orgGraph = get_graph_data(rdf_org_graph, None, None, emb_dim=emb_dim, org_only=True)
    
        print("ORIGINAL GRAPH STATISTICS")
        print(f"num Nodes = {self.orgGraph.num_nodes}")
        print(f"num Relations = {len(self.orgGraph.relations.keys())}")
        print(f"num Classes = {self.num_classes}")

        # init original graph training data
        g_idx, g_labels = self.get_idx_labels(self.orgGraph, self.sumGraphs[0].org2type)
        X_train, X_test, y_train, y_test = train_test_split(g_idx, g_labels,  test_size=0.2, random_state=1) 
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

        self.orgGraph.x_train = torch.tensor(X_train, dtype = torch.long)
        self.orgGraph.training_data.x_test = torch.tensor(X_test) 
        self.orgGraph.training_data.x_val = torch.tensor(X_val, dtype = torch.long)
        self.orgGraph.training_data.y_val = torch.tensor(y_val, dtype = torch.long)
        self.orgGraph.training_data.y_train = torch.tensor(y_train, dtype = torch.long)
        self.orgGraph.training_data.y_test = torch.tensor(y_test)

        # remove test data from summary graph data before making summary graph training data
        self.remove_test_data(X_test)

        # init summary graph data
        for sGraph in self.sumGraphs:
            sg_idx, sg_labels = self.get_idx_labels(sGraph, sGraph.sum2type)
            sGraph.training_data.x_train = torch.tensor(sg_idx, dtype = torch.long)
            sGraph.training_data.y_train = torch.tensor(sg_labels)
            
            print("SUMMARY GRAPH STATISTICS")
            print(f"num Nodes = {sGraph.num_nodes}")
            print(f"num Relations= {len(sGraph.relations.keys())}")

            # Assertion error: if more relations in summary graph than in original graph
            assert len(sGraph.relations.keys()) ==  len(self.orgGraph.relations.keys()), 'number of relations in summary graph and original graph differ'