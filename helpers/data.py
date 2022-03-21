from torch_geometric.datasets.entities import Entities

class Datasets:
    def __init__(self):
        self.datasets = {}
    
    def load_data(self, datasets):
        for dataset in datasets: 
            dataobject = Dataobject(name=dataset)
            dataobject.data = Entities(root='../data/', name=dataset)
            self.datasets[dataset] = dataobject

class Dataobject:
    '''
    name: str  
    data: pytorch geometric object
    '''
    def __init__(self, name, graphs):
        self.name = name
        self.graphs = graphs

class Graphs:
    # todo: edges, nodes etc
    def __init__(self, edge_index, edge_type, node_to_enum, num_nodes, nodes):
        self.edge_index = edge_index
        self.edge_type  = edge_type
        self.node_to_enum = node_to_enum
        self.num_nodes = num_nodes
        self.nodes = nodes

dataset_names = ['AIFB', 'MUTAG', 'AM']
datasets = Datasets.load_data(dataset_names)
