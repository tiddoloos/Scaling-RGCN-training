import argparse
from collections import defaultdict
from typing import Dict, List
from random import randint

def create_dummy_sum_map(path: str, sum_path: str, map_path: str, dataset: str, n_sumNodes: int) -> None:
    passed_nodes: set = set()
    random_orgNode_to_sumNode: Dict[str, str] = defaultdict()
    with open(path, 'r') as file:
        lines = file.read().splitlines()
        for triple in lines:
            triple_list = triple[:-2].split(" ", maxsplit=2)
            if triple_list != ['']:
                s, _, o = triple_list[0], triple_list[1], triple_list[2]
                for node in [s, o]:
                    if node not in passed_nodes:
                        passed_nodes.add(node)
                        sumNode = randint(0, n_sumNodes)
                        random_orgNode_to_sumNode[node] = sumNode
  
        write_sum_map_files(random_orgNode_to_sumNode, lines, f'{sum_path}{dataset}_sum_random{n_sumNodes}.nt', f'{map_path}{dataset}_map_random{n_sumNodes}.nt')

def write_sum_map_files(random_orgNode_to_sumNode: Dict[str, str],  lines: List[str], sum_path: str, map_path: str) -> None:
    # create sum file
    with open(sum_path, "w") as f:
        for triple in lines:
            triple_list = triple[:-2].split(" ", maxsplit=2)
            if triple_list != ['']:
                s, p, o = triple_list[0], triple_list[1], triple_list[2]
                obj = random_orgNode_to_sumNode[o]
                sub = random_orgNode_to_sumNode[s]
                f.write(f'<{sub}> {p} <{obj}> .\n')

     # create map file
    with open(map_path, "w") as m:
        for o_node, s_node in random_orgNode_to_sumNode.items():
            m.write(f'<{s_node}> <isSummaryOf> {str(o_node)} .\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='experiment arguments')
    parser.add_argument('-dataset', type=str, choices=['AIFB', 'AM', 'BGS', 'MUTAG', 'TEST2', 'TEST'], help='inidcate dataset name')
    parser.add_argument('-n', type=int, default=100)
    dataset = vars(parser.parse_args())['dataset']
    n_sumNodes = vars(parser.parse_args())['n']

    path = f'./{dataset}/{dataset}_complete.nt'
    sum_path = f'./{dataset}/dummy/sum/'
    map_path = f'./{dataset}/dummy/map/'

    create_dummy_sum_map(path, sum_path, map_path, dataset, n_sumNodes)