import argparse
import hashlib
import rdflib
import rdflib.term
import pathlib

from rdflib import URIRef
from typing import Callable, Dict


def check_blank(node: rdflib.term):
    if type(node) == rdflib.term.BNode:
        node = URIRef('http://example.org/'+ str(node))
    return node

def forward(node: rdflib.term, graph: rdflib.Graph, sum_type = 'out') -> str:
    sorted_preds = sorted(list(graph.predicates(subject=node)))
    hash = hashlib.sha1(','.join(sorted_preds).encode('utf8'))
    value = hash.hexdigest()
    node_id = 'sumnode:' + value
    return node_id, sum_type

def backward(node: rdflib.term, graph: rdflib.Graph, sum_type = 'in') -> str:
    sorted_preds = sorted(list(graph.predicates(object=node)))
    incoming_hash = hashlib.sha1(','.join(sorted_preds).encode('utf8'))
    value = incoming_hash.hexdigest()
    node_id = 'sumnode:' + value
    return node_id, sum_type

def forward_backward(node: rdflib.term, graph: rdflib.Graph, sum_type: str ='in_out') -> str:
    sorted_preds = sorted(list(graph.predicates(subject=node)))
    incoming_hash = hashlib.sha1(','.join(sorted_preds).encode('utf8'))
    sorted_outgoing_preds = sorted(list(graph.predicates(object=node)))
    outgoing_hash = hashlib.sha1(','.join(sorted_outgoing_preds).encode('utf8'))
    value = incoming_hash.hexdigest() + "-" + outgoing_hash.hexdigest()
    node_id = 'sumnode:' + value
    return node_id, sum_type

def create_sum_map(path: pathlib.Path, sum_path: pathlib.Path, map_path: pathlib.Path, format: str, id_creator: Callable[[URIRef, rdflib.Graph], str]) -> None:
    g = rdflib.Graph()
    sumGraph = rdflib.Graph()
    mapGraph = rdflib.Graph()

    with open(path, 'rb') as data:
        g.parse(data, format = format)
    mapping: Dict[str, str] = dict()

    # create sum graph
    for s_, p_, o_ in g:
        if type(o_) == rdflib.term.Literal:
            o_ = URIRef('http://example.org/literal')
        s_ = check_blank(s_)
        o_ = check_blank(o_)
        s, p, o = str(s_), str(p_), str(o_)
        for node, node_str in [(s_, s), (o_, o)]:
            if node_str not in mapping:                
                mapping[node_str], sum_type = id_creator(node, g)
        sum_triple = URIRef(mapping[s]), URIRef(p), URIRef(mapping[o])
        sumGraph.add(sum_triple)
    sumGraph.serialize(destination=f'{sum_path}{sum_type}.nt', format='nt')

    # create mappping as graph
    for node, sumNode in mapping.items(): 
        map_triple = URIRef(sumNode), URIRef('http://issummaryof'), URIRef(node)
        mapGraph.add(map_triple)
    mapGraph.serialize(destination=f'{map_path}{sum_type}.nt', format='nt')

parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'MUTAG', 'AM', 'TEST'], help='inidcate dataset name')
dataset = vars(parser.parse_args())['dataset']

path = f'./{dataset}/{dataset}_complete.nt'
sum_path = f'./{dataset}/attr/sum/{dataset}_sum_'
map_path = f'./{dataset}/attr/map/{dataset}_map_'
format = path.split('.')[-1]

create_sum_map(pathlib.Path(path), pathlib.Path(sum_path), pathlib.Path(map_path), format, forward_backward)
create_sum_map(pathlib.Path(path), pathlib.Path(sum_path), pathlib.Path(map_path), format, forward)
create_sum_map(pathlib.Path(path), pathlib.Path(sum_path), pathlib.Path(map_path), format, backward)
