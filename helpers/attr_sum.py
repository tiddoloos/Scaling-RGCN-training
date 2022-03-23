from rdflib import Graph
from pandas import DataFrame
from rdflib.plugins.sparql.processor import SPARQLResult
import yaml

input_format = '.nt'
output_format = '.csv'
input_names = {'AIFB': 'aifb_stripped'}
output_names = {'AIFB': 'aifb_attr_sum'}

def make_graph_from_nt(input_data):
    g = Graph()
    data = open(input_data, "rb")
    g.parse(data, format="nt")
    return g

def get_query(type):
    with open('helpers/SPARQL/attribute.yml') as file:
        queries = yaml.load(file, Loader=yaml.FullLoader)
        return queries[type]   

def make_attr_sum(graph):
    # was there in allessndro's SPARQL paper
    # PREFIX rdf: <http ://www.w3. org/1999/02/22=rdf=syntax=ns#>
    # found PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> in paper pf Campinas
    query = get_query('summary')
    sum_graph = graph.query(query)
    return sum_graph

def sparql_results_to_df(results: SPARQLResult) -> DataFrame:
    return DataFrame(
        data=([None if x is None else x.toPython() for x in row] for row in results))

def main_attr_sum(name):
    filepath = f'../data/AIFB/{input_names[name]}{input_format}'
    print(filepath)
    graph = make_graph_from_nt(filepath)
    sum_graph = make_attr_sum(graph)
    df = sparql_results_to_df(sum_graph)
    print(df)
    df.to_csv(f'../data/AIFB/{output_names[name]}{output_format}')

