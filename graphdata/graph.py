from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Graph:
    node_to_enum: Dict[str, int]
    num_nodes: int
    nodes: List[str]
    relations: Dict[str, int]
    orgNode2sumNode_dict: Dict[str, List[str]]
    sumNode2orgNode_dict: Dict[str, List[str]]
    org2type_dict: Dict[str, List[str]]
    org2type: Dict[str, List[str]]
    sum2type: Dict[str, List[str]]
    training_data: Any

    def get_training_data(self):
        return self.training_data
    