from utils import process_rdf_graph
from createMappings import main_map_org_nodes
import pandas as pd
from collections import defaultdict
import torch
from collections import defaultdict
from torch_geometric.data import Data
import torch
from sklearn.model_selection import train_test_split

class Graph:
    def __init__(self, edge_index, edge_type, node_to_enum, num_nodes, nodes):
        self.edge_index = edge_index
        self.edge_type  = edge_type
        self.node_to_enum = node_to_enum
        self.num_nodes = num_nodes
        self.nodes = nodes

class Dataset:
    graph_paths = {'AIFB': '../data/AIFB/AIFB_complete.n3'}
    sum_graph_paths = {'AIFB': '../data/AIFB/AIFB_attr_sum.n3'}
    def __init__(self, name: str(),
                sum2orgNodes_dict: dict(),
                sum2type_dict: dict(),
                org2type_dict: dict(), 
                dfSum2type: pd.DataFrame(), 
                typeLabels: dict(), 
                sumNodeLabels: dict(),
                num_labels: int()):

        self.name = name
        self.sum2orgNode = sum2orgNodes_dict
        self.sum2type = sum2type_dict
        self.org2type = org2type_dict
        self.dfSum2type = dfSum2type
        self.typeLabels = typeLabels
        self.sumNodeLabels = sumNodeLabels
        self.num_labels = num_labels
        self.sumGraph = None
        self.orgGraph = None
        self.sum_training_data = None
        self.org_training_data = None
    
    def get_idx_labels(self, graph, dictionary):
        train_indices, train_labels = [], []
        node2type_vec_dict = defaultdict(list)

        for sumNode in sorted(list(dictionary.keys())):
            sorted_dict = dict(sorted(dictionary[sumNode].items()))
            node2type_vec_dict[sumNode] = sorted_dict.values()

        for node, labs in node2type_vec_dict.items():
            if sum(list(labs)) != 0.0 and graph.node_to_enum.get(node) is not None:
                train_indices.append(graph.node_to_enum[node])
                train_labels.append(list(labs))
        return train_indices, train_labels

    def get_training_data(self):
        sg_idx, sg_labels = self.get_idx_labels(self.sumGraph, self.sum2type)
      
        # sg_edge_index = self.sumGraph.edge_index

        self.sum_training_data = Data(edge_index = self.sumGraph.edge_index)
        self.sum_training_data.idx = torch.tensor(sg_idx, dtype = torch.long)
        self.sum_training_data.labels = torch.tensor(sg_labels)

        g_idx, g_labels = self.get_idx_labels(self.orgGraph, self.org2type)

        g_edge_index = self.orgGraph.edge_index

        self.org_training_data = Data(edge_index = g_edge_index)
        X_train, X_test, y_train, y_test = train_test_split(g_idx, g_labels, test_size = 0.20)

        self.org_training_data.x_train = torch.tensor(X_train, dtype = torch.long)
        self.org_training_data.x_test = torch.tensor(X_test)
        self.org_training_data.y_train = torch.tensor(y_train, dtype = torch.long)
        self.org_training_data.y_test = torch.tensor(y_test)
        self.org_training_data.x = torch.tensor(g_idx, dtype=torch.long)
        self.org_training_data.y = torch.tensor(g_labels)

        print("Statistic Datasets:")
        print("SUMMARY GRAPH")
        print(f"NUM_NODES = {self.sumGraph.num_nodes}")
        print("ORGINAL GRAPH")
        print(f"NUM_NODES = {self.orgGraph.num_nodes}")
        print(f"NUM CLASSES = {self.num_labels}")
        return

    def collect_graph_data(self, dataset):
        edge_index, edge_type, nodes_dict, length_sorted_nodes, sorted_nodes = process_rdf_graph(Dataset.sum_graph_paths[dataset])
        sumGraph = Graph(edge_index, edge_type, nodes_dict, length_sorted_nodes, sorted_nodes)

        edge_index, edge_type, nodes_dict, length_sorted_nodes, sorted_nodes = process_rdf_graph(Dataset.graph_paths[dataset])
        orgGraph = Graph(edge_index, edge_type, nodes_dict, length_sorted_nodes, sorted_nodes)
    
        self.sumGraph = sumGraph
        self.orgGraph = orgGraph
        self.get_training_data()

def init_data_object(dataset):
    sum2orgNodes_dict, sum2type_dict, org2type_dict, dfSum2type, typeLabels, sumNodesLabels, labels = main_map_org_nodes(dataset)
    graphData = Dataset(dataset, sum2orgNodes_dict, sum2type_dict, org2type_dict, dfSum2type, typeLabels, sumNodesLabels, len(labels))
    graphData.collect_graph_data(dataset)
    return graphData

graphdata = init_data_object(dataset='AIFB')
