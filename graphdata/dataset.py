from copy import deepcopy
from typing import Tuple, List, Dict
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import torch

from helpers import timing
from graphdata.graphProcessing import parse_graph_nt, nodes2type_mapping, get_node_mappings_dict, encode_org_node_labels, encode_sum_node_labels, remove_eval_data, get_idx_labels, get_classes
from graphdata.graph import Graph


class Dataset:
    def __init__(self, org_path: str, sum_path: str, map_path: str) -> None:
        self.org_path: str = org_path
        self.sum_path: str = sum_path
        self.map_path: str = map_path
        self.sumGraphs: list = []
        self.orgGraph: Graph = None
        self.enum_classes: Dict[str, int] = None
        self.num_classes: int = None

    def make_trainig_data(self):
        self.orgGraph.org2type  = encode_org_node_labels(self.orgGraph.org2type_dict, self.enum_classes, self.num_classes)

        g_idx, g_labels = get_idx_labels(self.orgGraph, self.orgGraph.org2type)
        X_train, X_test, y_train, y_test = train_test_split(g_idx, g_labels,  test_size=0.2, random_state=1) 
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

        self.orgGraph.training_data.x_train = torch.tensor(X_train, dtype = torch.long)
        self.orgGraph.training_data.x_test = torch.tensor(X_test) 
        self.orgGraph.training_data.x_val = torch.tensor(X_val, dtype = torch.long)
        self.orgGraph.training_data.y_val = torch.tensor(y_val, dtype = torch.long)
        self.orgGraph.training_data.y_train = torch.tensor(y_train, dtype = torch.long)
        self.orgGraph.training_data.y_test = torch.tensor(y_test)

        print("ORIGINAL GRAPH STATISTICS")
        print(f"file name = {self.orgGraph.name}")
        print(f"num Nodes = {self.orgGraph.num_nodes}")
        print(f"num Relations = {len(self.orgGraph.relations.keys())}")
        print(f"num Classes = {self.num_classes}")
        timing.log('ORGINAL GRPAH LOADED')

        # romeve evaluation data from org2type: we use org2type to create weighted labels for summary graph training
        to_remove = X_test + X_val
        org2type_pruned = remove_eval_data(to_remove, self.orgGraph)

        for sumGraph in self.sumGraphs:
            sumGraph.sum2type  = encode_sum_node_labels(sumGraph.sumNode2orgNode_dict, org2type_pruned, self.enum_classes, self.num_classes)

            sg_idx, sg_labels = get_idx_labels(sumGraph, sumGraph.sum2type)
            sumGraph.training_data.x_train = torch.tensor(sg_idx, dtype = torch.long)
            sumGraph.training_data.y_train = torch.tensor(sg_labels)
            
            print("SUMMARY GRAPH STATISTICS")
            print(f"file name = {sumGraph.name}")
            print(f"num Nodes = {sumGraph.num_nodes}")
            print(f"num Relations= {len(sumGraph.relations.keys())}")
            timing.log('SUMGRPAH LOADED')
            # Assertion: if more relations in summary graph than in original graph
            assert len(sumGraph.relations.keys()) ==  len(self.orgGraph.relations.keys()), 'number of relations in summary graph and original graph differ'
        
    def get_file_names(self) -> Tuple[List[str], List[str]]:
        sum_files = [f for f in listdir(self.sum_path) if not f.startswith('.') if isfile(join(self.sum_path, f))]
        map_files = [f for f in listdir(self.map_path) if not f.startswith('.') if isfile(join(self.map_path, f))]
        assert len(sum_files) == len(map_files), f'for every summary file there needs to be a map file.{sum_files} and {map_files}'
        return sorted(sum_files), sorted(map_files)

    def init_dataset(self) -> None:
        org_graph_triples = parse_graph_nt(self.org_path)

        classes = get_classes(org_graph_triples)
        enum_classes = {lab: i for i, lab in enumerate(classes)}
        self.enum_classes, self.num_classes = enum_classes, len(classes)
        
        org2type_dict = nodes2type_mapping(org_graph_triples, classes)

        file_name = self.org_path.split('/')[-1]
        self.orgGraph = Graph(file_name, deepcopy(org2type_dict))
        self.orgGraph.init_graph(org_graph_triples)

        # init summary graph data
        sum_files, map_files = self.get_file_names()
        for i, _ in enumerate(sum_files):
            sum_path = f'{self.sum_path}/{sum_files[i]}'
            map_path = f'{self.map_path}/{map_files[i]}'
            sum_graph_triples = parse_graph_nt(sum_path)
            map_graph_triples = parse_graph_nt(map_path)
            file_name = sum_path.split('/')[-1]
            sGraph = Graph(file_name, deepcopy(org2type_dict))
            sGraph.init_graph(sum_graph_triples)
            sGraph.orgNode2sumNode_dict, sGraph.sumNode2orgNode_dict = get_node_mappings_dict(map_graph_triples)
            self.sumGraphs.append(sGraph)

        self.make_trainig_data()
    