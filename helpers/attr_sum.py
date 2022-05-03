import yaml
from rdflib.graph import Graph
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.processor import SPARQLResult
from typing import Tuple
from pandas import DataFrame

# input_format = 'n3'
input_names = {'AIFB': './data/AIFB/aifb_complete.n3', 'MUTAG': './data/MUTAG/mutag_complete.nt'}
output_names = {'AIFB': 'aifb_attr_sum'}

def make_rdf_graph(file_path: str) -> Graph:
    input_format = file_path.split('.')[-1]
    g = Graph()
    data = open(file_path, 'rb')
    g.parse(data, format = input_format)
    return g

def apply_query(graph: Graph, key: str) -> Tuple[str, str, str]:
    with open('./helpers/SPARQL/attribute.yml') as file:
        queries = yaml.load(file, Loader=yaml.FullLoader)
    query = queries[key]
    result = graph.query(query)
    print(result.__dict__.keys())
    return result

def main_attr_sum(name: str) -> Graph:
    filepath = f'{input_names[name]}'
    graph = make_rdf_graph(filepath)
    sum_graph = apply_query(graph, 'summary')
    return sum_graph

# main_attr_sum('MUTAG')
sum_graph = main_attr_sum('AIFB')
# print(sum_graph._genbindings.__dict__.keys())
print(sum_graph.bindings)
# print(DataFrame(sum_graph, columns=sum_graph.vars))
# for s, p, o in sum_graph:
#     print(s, p , o)


#SPARQL wrapper
