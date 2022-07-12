import argparse
from collections import defaultdict
from typing import Dict, List
import mmh3

def create_sum_map(path: str, sum_path: str, map_path: str, dataset: str) -> None:
    outgoing_properties: Dict[str, str] = defaultdict(set)
    incoming_properties: Dict[str, str] = defaultdict(set)

    with open(path, 'r') as file:
        triples = file.read().splitlines()
        for triple in triples:
            triple_list = triple[:-2].split(" ", maxsplit=2)
            if triple_list != ['']:
                s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
                if p != '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
                    outgoing_properties[s].add(p)
                    if o.startswith("\""):
                        incoming_properties['http://example.org/literal'].add(p)
                    else:
                        incoming_properties[o].add(p)

        outgoing_properties_hashed: Dict[str, int] = dict()
        for s1, p1 in outgoing_properties.items():
            property_hash1 = mmh3.hash128(','.join(sorted(list(p1))).encode('utf8'))
            outgoing_properties_hashed[s1] = property_hash1

        incoming_properties_hashed: Dict[str, int] = dict()
        for s2, p2 in incoming_properties.items():
            property_hash2 = mmh3.hash128(','.join(sorted(list(p2))).encode('utf8'))
            incoming_properties_hashed[s2] = property_hash2
  
        incoming_and_outgoing_properties_hashed: Dict[str, int] = dict()
        for entity in set(incoming_properties.keys()).union(set(outgoing_properties.keys())):
            incoming = incoming_properties_hashed[entity] if entity in incoming_properties_hashed else 0
            outgoing = outgoing_properties_hashed[entity] if entity in outgoing_properties_hashed else 0
            combined_hash = incoming + outgoing
            incoming_and_outgoing_properties_hashed[entity] = combined_hash

        write_sum_map_files(outgoing_properties_hashed, triples, f'{sum_path}{dataset}_sum_out.nt', f'{map_path}{dataset}_map_out.nt')
        write_sum_map_files(incoming_properties_hashed, triples, f'{sum_path}{dataset}_sum_in.nt', f'{map_path}{dataset}_map_in.nt')
        write_sum_map_files(incoming_and_outgoing_properties_hashed, triples, f'{sum_path}{dataset}_sum_in_out.nt', f'{map_path}{dataset}_map_in_out.nt')

def write_sum_map_files(property_hashes: Dict[str, int], triples: List[str], sum_path: str, map_path: str) -> None:

    property_keys = property_hashes.keys()
    mapping: Dict[int, str] = dict()

    # create sum file
    with open(sum_path, "w") as f:
        for triple in triples:
            triple_list = triple[:-2].split(" ", maxsplit=2)
            if triple_list != ['']:
                s, p, o = triple_list[0].lower(), triple_list[1].lower(), triple_list[2].lower()
                if o.startswith("\"") and 'http://example.org/literal' in property_keys:
                    obj = property_hashes['http://example.org/literal']
                else:
                    obj = property_hashes[o] if o in property_keys else '0'
                sub = property_hashes[s] if s in property_keys else '0'
                mapping[s] = sub
                mapping[o] = obj
                f.write(f'<{sub}> {p} <{obj}> .\n')

    # create map file
    with open(map_path, "w") as m:
        for o_node, s_node in mapping.items():
            m.write(f'<{s_node}> <isSummaryOf> {str(o_node)} .\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='experiment arguments')
    parser.add_argument('-dataset', type=str, choices=['AIFB', 'AM', 'BGS', 'MUTAG'], help='inidcate dataset name')
    dataset = vars(parser.parse_args())['dataset']

    path = f'./{dataset}/{dataset}_complete.nt'
    sum_path = f'./{dataset}/attr/sum/'
    map_path = f'./{dataset}/attr/map/'

    create_sum_map(path, sum_path, map_path, dataset)
