import torch

from typing import Tuple, List, Dict
from os import listdir
from os.path import isfile, join
from torch import Tensor
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from graphdata.graphUtils import process_rdf_graph
from graphdata.createMapping import main_createMappings, encode_node_labels


class Graph:
    device = torch.device(str('cuda:0') if torch.cuda.is_available() else 'cpu')
    def __init__(self, edge_index: Tensor, edge_type: Tensor, node_to_enum: dict,
                num_nodes: int, nodes, relations_dict: dict, orgNode2sumNode_dict: dict, 
                sumNode2orgNode_dict: dict, org2type_dict: dict, org2type: dict, sum2type: dict) -> None:

        self.edge_index = edge_index.to(self.device)
        self.edge_type  = edge_type.to(self.device)
        self.node_to_enum = node_to_enum
        self.num_nodes = num_nodes
        self.nodes = nodes
        self.relations = relations_dict
        self.orgNode2sumNode_dict = orgNode2sumNode_dict
        self.sumNode2orgNode_dict = sumNode2orgNode_dict
        self.org2type_dict = org2type_dict
        self.org2type = org2type
        self.sum2type = sum2type
        self.training_data = None
        self.embedding = None

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
            #make a copy to preserve orginal data in the data opject
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
        return sorted(sum_files), sorted(map_files)

    def init_dataset(self) -> None:
        sum_files, map_files = self.get_file_names()
        assert len(sum_files) == len(map_files), f'for every summary file there needs to be a map file.{sum_files} and {map_files}'

        #collect summary graph data
        for i in range(len(sum_files)):
            sum_path = f'{self.sum_path}/{sum_files[i]}'
            map_path = f'{self.map_path}/{map_files[i]}'
            sum2type, org2type, self.enum_classes, self.num_classes, orgNode2sumNode_dict, sumNode2orgNode_dict, org2type_dict = main_createMappings(self.org_path, map_path)
            edge_index, edge_type, node_to_enum, length_sorted_nodes, sorted_nodes, relations_dict = process_rdf_graph(sum_path)
            
            sGraph = Graph(edge_index, edge_type, node_to_enum, length_sorted_nodes, sorted_nodes, relations_dict, orgNode2sumNode_dict, sumNode2orgNode_dict, org2type_dict, org2type, sum2type)
            self.sumGraphs.append(sGraph)

        edge_index, edge_type, node_to_enum, length_sorted_nodes, sorted_nodes, relations_dict = process_rdf_graph(self.org_path)
        self.orgGraph = Graph(edge_index, edge_type, node_to_enum, length_sorted_nodes, sorted_nodes, relations_dict, None, None, None, None, None)

        print("ORIGINAL GRAPH STATISTICS")
        print(f"num Nodes = {self.orgGraph.num_nodes}")
        print(f"num Relations = {len(self.orgGraph.relations.keys())}")
        print(f"num Classes = {self.num_classes}")

        g_idx, g_labels = self.get_idx_labels(self.orgGraph, self.sumGraphs[0].org2type)
        X_train, X_test, y_train, y_test = train_test_split(g_idx, g_labels,  test_size=0.2, random_state=1) 
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

        self.orgGraph.training_data = Data(edge_index = self.orgGraph.edge_index)
        self.orgGraph.training_data.x_train = torch.tensor(X_train, dtype = torch.long)
        self.orgGraph.training_data.x_test = torch.tensor(X_test) 
        self.orgGraph.training_data.x_val = torch.tensor(X_val, dtype = torch.long)
        self.orgGraph.training_data.y_val = torch.tensor(y_val, dtype = torch.long)
        self.orgGraph.training_data.y_train = torch.tensor(y_train, dtype = torch.long)
        self.orgGraph.training_data.y_test = torch.tensor(y_test)

        # remove test data from summary nodes to types before making summary graph training data
        self.remove_test_data(X_test)

        # get training data of summary graphs
        for sGraph in self.sumGraphs:
            sg_idx, sg_labels = self.get_idx_labels(sGraph, sGraph.sum2type)
            sGraph.training_data = Data(edge_index = sGraph.edge_index)
            sGraph.training_data.x_train = torch.tensor(sg_idx, dtype = torch.long)
            sGraph.training_data.y_train = torch.tensor(sg_labels)
            
            print("SUMMARY GRAPH STATISTICS")
            print(f"num Nodes = {sGraph.num_nodes}")
            print(f"num Relations= {len(sGraph.relations.keys())}")

            # if more relations in summary graph than in original graph -> assert
            assert len(sGraph.relations.keys()) ==  len(self.orgGraph.relations.keys()), 'number of relations in summary graph and original graph differ'