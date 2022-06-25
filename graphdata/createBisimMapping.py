import argparse
import csv

from collections import defaultdict
from email.policy import default
from os import listdir
from typing import Dict, List


"""Run this file from './graphdata/' .
This file creates a mapping of the (k)bisimualition output created with the 
BiSimulation pipeline of Till Blume: https://github.com/t-blume/fluid-spark.
For each folder in <dataset>/bisim/bisimOutput, triples like 'sumNode isSummaryOf orgNode'
are stored in a .nt file in <dataset>/bisim/map/ .
"""


def csv_to_mapping(path: str, key_idx: int, value_idx: int) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    with open(path, 'rt') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            mapping[line[key_idx]].append(line[value_idx])
    return mapping

def write_to_nt(orgHash_to_orgNode: defaultdict(list), sumNode_to_orgHash: defaultdict(list), map_path: str, k: str) -> None:
    with open(f'{map_path}{k}.nt', 'w') as m:
            for sumNode, orgHashes in sumNode_to_orgHash.items():
                for orgHash in orgHashes:
                    nodes = orgHash_to_orgNode[orgHash]
                    for node in nodes:
                        m.write(f'<{sumNode}> <isSummaryOf> <{node}> .\n')

def create_bisim_map_nt(path: str, map_path: str) -> None:
    dirs = sorted([x for x in listdir(path) if not x.startswith('.')])
    for dir in dirs:
        files = sorted([s for s in listdir(f'{path}/{dir}/') if not s.startswith('.')])
        for file in files:
            if file.startswith('orgNode'):
                orgHash_to_orgNode = csv_to_mapping(f'{path}/{dir}/{file}', 1, 0)
            else:
                sumNode_to_orgHash = csv_to_mapping(f'{path}/{dir}/{file}', 0, 1)
        k = dir.split('_')[-1]
        write_to_nt(orgHash_to_orgNode, sumNode_to_orgHash, map_path, k)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='experiment arguments')
    parser.add_argument('-dataset', type=str, choices=['AIFB', 'AM', 'MUTAG', 'TEST'], help='inidcate dataset name')
    dataset = vars(parser.parse_args())['dataset']

    path = f'./{dataset}/bisim/bisimOutput'
    map_path = f'./{dataset}/bisim/map/AIFB_bisim_map_'

    create_bisim_map_nt(path, map_path)
