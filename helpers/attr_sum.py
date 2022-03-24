from rdflib.graph import Graph
from rdflib.plugins.sparql import prepareQuery
from pandas import DataFrame
from rdflib.plugins.sparql.processor import SPARQLResult
import yaml

input_format = 'n3'
input_names = {'AIFB': 'aifb_fixed_complete'}
output_names = {'AIFB': 'aifb_attr_sum'}

def make_rdf_graph(file_path):
    g = Graph()
    data = open(file_path, 'rb')
    g.parse(data, format = input_format)
    return g

def apply_query(graph, key):
    with open('helpers/SPARQL/attribute.yml') as file:
        queries = yaml.load(file, Loader=yaml.FullLoader)
    query = prepareQuery(queries[key])
    result = graph.query(query)
    return result

def main_attr_sum(name):
    filepath = f'../data/AIFB/{input_names[name]}.{input_format}'
    graph = make_rdf_graph(filepath)
    sum_graph = apply_query(graph, 'summary')
    return sum_graph



