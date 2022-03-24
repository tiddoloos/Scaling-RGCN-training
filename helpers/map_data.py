from pickle import TRUE
from attr_sum import make_rdf_graph
import csv
from collections import defaultdict
import pandas as pd
from csv import reader
import ast

output_format = 'csv'

def map_org_labels(dataset_name):
    rel = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    org_file = f'../data/{dataset_name}/{dataset_name}_complete.n3'
    org_graph = make_rdf_graph(org_file)
    file_to_write = open(f'../data/{dataset_name}/mappings/{dataset_name}_mapped_org_labels.{output_format}', 'w')
    writer = csv.writer(file_to_write)

    node_to_types = defaultdict(list)
    labels = []
    for s, p, o in org_graph:
        # this if statement makes sure all nodes are entity nodes
        if str(p).lower() == rel.lower() and str(s).split('#')[0] != 'http://swrc.ontoware.org/ontology':
            s_ = str(s).lower()
            type_ = str(o).lower()
            node_to_types[s_].append(type_)
            labels.append(type_)
    labels = sorted(list(set(labels)))

    # for each nodes_to_types write the types
    for node, type_list in node_to_types.items():
        sorted_type_list = sorted(type_list)
        writer.writerow([node, sorted_type_list])
    return labels

def create_map_csv(dataset_name, labels):
    column_names = ['node'] + labels
    df = pd.DataFrame(columns=column_names)
    with open(f'../data/{dataset_name}/mappings/{dataset_name}_mapped_org_labels.{output_format}', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            map = dict.fromkeys(column_names, 0)
            # row variable is a list that represents a row in csv
            map['node'] = row[0]
            entities = ast.literal_eval(row[1])
            for entity in entities:
                map[entity] = 1
            df = pd.concat([df, pd.DataFrame.from_records([map])])
    df.set_index('node', inplace = True)
    df.groupby(level=0).sum()
    df.to_csv(f'../data/{dataset_name}/mappings/{dataset_name}_mapped_org_labels_vector.{output_format}',)
            
labels = map_org_labels('AIFB')
create_map_csv('AIFB', labels)
