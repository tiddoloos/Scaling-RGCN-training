from copy import deepcopy
from typing import Tuple, List
from os import listdir
from os.path import isfile, join

from graphdata.graphProcessing import make_rdf_graph, nodes2type_mapping
from graphdata.graph import get_graph_data, make_graph_trainig_data


class Dataset:
    def __init__(self, name: str) -> None:
        self.org_path = f'./graphdata/{name}/{name}_complete.nt'
        self.sum_path = f'./graphdata/{name}/attr/sum/'
        self.map_path = f'./graphdata/{name}/attr/map/'
        self.sumGraphs = []
        self.orgGraph = None
        self.enum_classes = None
        self.num_classes = None

    def get_file_names(self) -> Tuple[List[str], List[str]]:
        sum_files = [f for f in listdir(self.sum_path) if not f.startswith('.') if isfile(join(self.sum_path, f))]
        map_files = [f for f in listdir(self.map_path) if not f.startswith('.') if isfile(join(self.map_path, f))]
        assert len(sum_files) == len(map_files), f'for every summary file there needs to be a map file.{sum_files} and {map_files}'
        return sorted(sum_files), sorted(map_files)

    def init_dataset(self, emb_dim: int) -> None:
        rdf_org_graph = make_rdf_graph(self.org_path)
        classes, org2type_dict = nodes2type_mapping(rdf_org_graph)
        enum_classes = {lab: i for i, lab in enumerate(classes)}
        self.enum_classes, self.num_classes = enum_classes, len(classes)

        sum_files, map_files = self.get_file_names()

        # init summary graph data
        for i in range(len(sum_files)):
            sum_path = f'{self.sum_path}/{sum_files[i]}'
            map_path = f'{self.map_path}/{map_files[i]}'
            rdf_sum_graph = make_rdf_graph(sum_path)
            rdf_map_graph = make_rdf_graph(map_path)
            sGraph = get_graph_data(rdf_org_graph, rdf_sum_graph, rdf_map_graph, emb_dim, deepcopy(org2type_dict), self.enum_classes, self.num_classes, org=False)
            self.sumGraphs.append(sGraph)

        # init original graph data
        self.orgGraph = get_graph_data(rdf_org_graph, rdf_sum_graph, rdf_map_graph, emb_dim, deepcopy(org2type_dict), self.enum_classes, self.num_classes, org=True)
        
        make_graph_trainig_data(self.orgGraph, self.sumGraphs, self.enum_classes, self.num_classes)