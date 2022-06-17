import csv

from collections import defaultdict
from os import listdir
from os.path import isfile, join
from typing import Dict

def csv_to_mapping(path: str, key_idx, value_idx):
    mapping: Dict[str, str] = defaultdict(list)
    with open(path, 'rt')as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            mapping[line[key_idx]].append(line[value_idx])
    return mapping

def main_create_bisim_mapping(path: str, map_path: str):
    files = [f for f in listdir(path)if not f.startswith('.') if isfile(join(path, f))]
    for file in files:
        if file.startswith('orgNode'):
            orgHash_to_orgNode = csv_to_mapping(f'{path}{file}', 1, 0)
        else:
            sumNode_to_orgHash = csv_to_mapping(f'{path}{file}', 0, 1)

    oH_2_oH_keys = orgHash_to_orgNode.keys()
    with open(map_path, "w") as m:
        for sumNode, orgHashes in sumNode_to_orgHash.items():
            for orgHash in orgHashes:
                nodes = orgHash_to_orgNode[orgHash]
                for node in nodes:
                    m.write(f'<{sumNode}> <isSummaryOf> <{node}> .\n')




path = './AIFB_BisimTestPipeline_k1/'
map_path = './map/AIFB_map_k1.nt'
main_create_bisim_mapping(path, map_path)

