import rdflib 
import pathlib
from typing import Any, Callable, Dict
import hashlib
import rdflib.term
from rdflib import URIRef


def create_sum_mapping(path: pathlib.Path, sum_path: pathlib.Path, map_path: pathlib.Path, format: str, id_creator: Callable[[rdflib.term.IdentifiedNode, rdflib.Graph], str]) -> None:
    g = rdflib.Graph()
    sumGraph = rdflib.Graph()
    mapGraph = rdflib.Graph()

    with open(path, 'rb') as data:
        g.parse(data, format = format)
    mapping: Dict[str, str] = dict()

    for s_, p_, o_ in g:
        s, p, o = str(s_), str(p_), str(o_)
        for node, node_str in [(s_, s), (o_, o)]:
            if node_str not in mapping:                
                mapping[node_str], sum_type = id_creator(node, g)
        sum_triple = URIRef(mapping[s]), URIRef(p), URIRef(mapping[o])
        sumGraph.add(sum_triple)
    sumGraph.serialize(destination=f'{sum_path}{sum_type}.nt', format='nt')

    map_p = 'http://issummaryof'
    for node, sumNode in mapping.items():
        map_triple = URIRef(sumNode), URIRef(map_p), URIRef(node)
        mapGraph.add(map_triple)
    mapGraph.serialize(destination=f'{map_path}{sum_type}.nt', format='nt')


def forward(node: rdflib.term.IdentifiedNode, graph: rdflib.Graph, sum_type = 'forw') -> str:
    sorted_preds = sorted(list(graph.predicates(subject=node)))
    hash = hashlib.sha1(','.join(sorted_preds).encode('utf8'))
    value = hash.hexdigest()
    node_id = 'sumnode:' + value
    return node_id, sum_type

def forward_backward(node: rdflib.term.IdentifiedNode, graph: rdflib.Graph, sum_type = 'forw_back') -> str:
    sorted_preds = sorted(list(graph.predicates(subject=node)))
    incoming_hash = hashlib.sha1(','.join(sorted_preds).encode('utf8'))
    sorted_outgoing_preds = sorted(list(graph.predicates(object=node)))
    outgoing_hash = hashlib.sha1(','.join(sorted_outgoing_preds).encode('utf8'))
    value = incoming_hash.hexdigest() + "-" + outgoing_hash.hexdigest()
    node_id = 'sumnode:' + value
    return node_id, sum_type


path = './data/TEST/TEST_complete.nt'
sum_path = './data/TEST/attr/sum/TEST_sum_'
map_path = './data/TEST/attr/map/TEST_map_'
format = path.split('.')[-1]
create_sum_mapping(pathlib.Path(path), pathlib.Path(sum_path), pathlib.Path(map_path), format, forward_backward)
create_sum_mapping(pathlib.Path(path), pathlib.Path(sum_path), pathlib.Path(map_path), format, forward)