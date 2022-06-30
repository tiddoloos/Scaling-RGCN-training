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

def reformat(node):
    if node.endswith('#string'):
        split = node.rsplit('^^', 1)
        if len(split) < 2:
           split.insert(0,'""')
           node = '^^<'.join(split) + '>'
           return node
        else:
            string = '"' + split[0] + '"'
            lit = '<' + split[1] + '>'
            node = '^^'.join([string, lit])
            return node
    if node.startswith('http://informatik.uni-kiel.de/fluid#'):
        node = node.replace('http://informatik.uni-kiel.de/fluid#', '_:')
        return node
    else:
        node = '<' + node + '>'
        return node

def csv_to_mapping(path: str, org: bool = True) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    with open(path, 'rt') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            line = ','.join(line)
            line = line.rsplit(',', 1)
            if org:
                node = reformat(line[0])
                mapping[line[1]].append(node)
            else:
                mapping[line[0]].append(line[1])
    return mapping

def write_to_nt(orgHash_to_orgNode: defaultdict(list), sumNode_to_orgHash: defaultdict(list), map_path: str, k: str) -> None:
    with open(f'{map_path}{k}.nt', 'w') as m:
            for sumNode, orgHashes in sumNode_to_orgHash.items():
                for orgHash in orgHashes:
                    nodes = orgHash_to_orgNode[orgHash]
                    for node in nodes:
                        m.write(f'<{sumNode}> <isSummaryOf> {node} .\n')

def create_bisim_map_nt(path: str, map_path: str) -> None:
    # nodes_set = set()
    # with open('AIFB/AIFB_complete.nt', 'r') as file:
    #     triples = file.read().replace(' .', '').splitlines()
    #     for triple in triples:
    #         triple_list = triple.split(" ", maxsplit=2)
    #         if triple_list != ['']:
    #             s, p, o = triple_list[0], triple_list[1], triple_list[2]
    #             nodes_set.add(s.lower())
    #             nodes_set.add(o.lower())

    dirs = sorted([x for x in listdir(path) if not x.startswith('.')])
    for dir in dirs:
        files = sorted([s for s in listdir(f'{path}/{dir}/') if not s.startswith('.')])
        for file in files:
            if file.startswith('orgNode'):
                orgHash_to_orgNode = csv_to_mapping(f'{path}/{dir}/{file}')
            else:
                sumNode_to_orgHash = csv_to_mapping(f'{path}/{dir}/{file}', org=False)
        k = dir.split('_')[-1]
        write_to_nt(orgHash_to_orgNode, sumNode_to_orgHash, map_path, k)
    # print(orgHash_to_orgNode.keys())
    # # pint()
    # no_overlap = 0
    # overlap = 0
    # for node in orgHash_to_orgNode.keys():
    #     if node not in nodes_set and node.startswith("\"") == False:
    #         # print(node)
    #         no_overlap += 1
    #     else:
    #         overlap += 1
    # print(no_overlap)
    # print(overlap)
    #         else:
    #             sumNode_to_orgHash = csv_to_mapping(f'{path}/{dir}/{file}', 0, 1)
    #     k = dir.split('_')[-1]
    #     write_to_nt(orgHash_to_orgNode, sumNode_to_orgHash, map_path, k)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='experiment arguments')
    parser.add_argument('-dataset', type=str, choices=['AIFB', 'AM', 'MUTAG', 'TEST'], help='inidcate dataset name')
    dataset = vars(parser.parse_args())['dataset']

    path = f'./{dataset}/bisim/bisimOutput'
    map_path = f'./{dataset}/bisim/map/AIFB_bisim_map_'

    create_bisim_map_nt(path, map_path)
