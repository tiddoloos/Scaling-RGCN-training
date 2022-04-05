from helpers.utils import process_rdf_graph
# from helpers.createMappings import main_createMappings
from helpers.clean_createMapping import main_createMappings
from collections import defaultdict
import torch
from collections import defaultdict
from torch_geometric.data import Data
import torch
from sklearn.model_selection import train_test_split

class Graph:
    def __init__(self, edge_index, edge_type, node_to_enum, num_nodes, nodes, relations_dict):
        self.edge_index = edge_index
        self.edge_type  = edge_type
        self.node_to_enum = node_to_enum
        self.num_nodes = num_nodes
        self.nodes = nodes
        self.relations = relations_dict

class Dataset:
    graph_paths = {'AIFB': 'data/AIFB/AIFB_complete.n3'}
    sum_graph_paths = {'AIFB': 'data/AIFB/AIFB_attr_sum.n3'}

    def __init__(self, name: str()):
        self.name = name
        self.orgNode2sumNode_dict = None
        self.sumNode2orgNode_dict = None
        self.org2type = None
        self.sum2type = None
        self.enum_classes = None
        self.num_classes = None
        self.sumGraph = None
        self.orgGraph = None
        self.sum_training_data = None
        self.org_training_data = None
    
    def get_idx_labels(self, graph, dictionary):
        train_indices, train_labels = [], []
        for node, labs in dictionary.items():
            if sum(list(labs)) != 0.0 and graph.node_to_enum.get(node) is not None:
                train_indices.append(graph.node_to_enum[node])
                train_labels.append(list(labs))
        return train_indices, train_labels

    def make_training_data(self, isOrg=False):
        if isOrg:
            g_idx, g_labels = self.get_idx_labels(self.orgGraph, self.org2type)
            X_train, X_test, y_train, y_test = train_test_split(g_idx, g_labels,  test_size=0.2, random_state=1) 
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

            self.org_training_data = Data(edge_index = self.orgGraph.edge_index)    
            self.org_training_data.x_train = torch.tensor(X_train, dtype = torch.long)
            self.org_training_data.x_test = torch.tensor(X_test)
            self.org_training_data.x_val = torch.tensor(X_val, dtype = torch.long)
            self.org_training_data.y_val = torch.tensor(y_val, dtype = torch.long)
            self.org_training_data.y_train = torch.tensor(y_train, dtype = torch.long)
            self.org_training_data.y_test = torch.tensor(y_test)
            self.org_training_data.idx = torch.tensor(g_idx, dtype=torch.long)
            self.org_training_data.y = torch.tensor(g_labels)

        #remove test data and then make summary.
        else:
            sg_idx, sg_labels = self.get_idx_labels(self.sumGraph, self.sum2type)
            self.sum_training_data = Data(edge_index = self.sumGraph.edge_index)
            self.sum_training_data.idx = torch.tensor(sg_idx, dtype = torch.long)
            self.sum_training_data.y = torch.tensor(sg_labels)

            print("Statistic Datasets:")
            print("SUMMARY GRAPH")
            print(f"NUM_NODES = {self.sumGraph.num_nodes}")
            print(f"NUM_RELATIONS = {len(self.sumGraph.relations.keys())}")
            print("ORGINAL GRAPH")
            print(f"NUM_NODES = {self.orgGraph.num_nodes}")
            print(f"NUM_RELATIONS = {len(self.orgGraph.relations.keys())}")
            print(f"NUM CLASSES = {self.num_classes}")
            
        return

    def collect_graph_data(self):
        edge_index, edge_type, nodes_dict, length_sorted_nodes, sorted_nodes, relations_dict = process_rdf_graph(self.graph_paths[self.name])
        orgGraph = Graph(edge_index, edge_type, nodes_dict, length_sorted_nodes, sorted_nodes, relations_dict)
        self.orgGraph = orgGraph
        self.make_training_data(isOrg=True)

        # get test data en remove from graph
        # make summarized graph

        edge_index, edge_type, nodes_dict, length_sorted_nodes, sorted_nodes, relations_dict = process_rdf_graph(self.sum_graph_paths[self.name])
        sumGraph = Graph(edge_index, edge_type, nodes_dict, length_sorted_nodes, sorted_nodes, relations_dict)
        self.sumGraph = sumGraph
        
        self.make_training_data()

    def init_dataset(self):
        self.sum2type, self.org2type, self.enum_classes, self.num_classes , self.orgNode2sumNode_dict, self.sumNode2orgNode_dict = main_createMappings(self.name)
        self.collect_graph_data()
