from attr_sum import make_rdf_graph
import csv
from collections import defaultdict
import pandas as pd
from csv import reader
import ast
from collections import defaultdict

output_format = 'csv'

def write_to_csv(path, dict_object):
    file_to_write = open(path, 'w')
    writer = csv.writer(file_to_write)
    # for each nodes_to_types write the types
    for node, type_list in dict_object.items():
        sorted_type_list = sorted(type_list)
        writer.writerow([node, sorted_type_list])

#map_original nodes
def map_org_labels(dataset_name):
    rel = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    org_file = f'../data/{dataset_name}/{dataset_name}_complete.n3'
    org_graph = make_rdf_graph(org_file)

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
    write_to_csv(f'../data/{dataset_name}/mappings/{dataset_name}_mapped_org_labels.{output_format}', node_to_types)
    return labels

#save orginal nodes mapping to pd.DataFrame
def create_map_csv(dataset_name, labels):
    column_names = ['node'] + labels
    dfOrg2Type = pd.DataFrame(columns=column_names)
    with open(f'../data/{dataset_name}/mappings/{dataset_name}_mapped_org_labels.{output_format}', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            map = dict.fromkeys(column_names, 0.0)
            # row variable is a list that represents a row in csv
            map['node'] = row[0]
            entities = ast.literal_eval(row[1])
            for entity in entities:
                map[entity] = 1.0
            dfOrg2Type = pd.concat([dfOrg2Type, pd.DataFrame.from_records([map])])
    dfOrg2Type.set_index('node', inplace = True)
    dfOrg2Type.groupby(level=0).sum()
    dfOrg2Type.to_csv(f'../data/{dataset_name}/mappings/{dataset_name}_mapped_org2type.{output_format}')
    return dfOrg2Type

#create summary nodes mapping
def map_sum_nodes(dataset_name, dfOrg2Type, labels):
    sum_map_file = f'../data/{dataset_name}/{dataset_name}_attr_map.n3'
    map_graph = make_rdf_graph(sum_map_file)
    sum2orgNodes_dict = defaultdict(list)
    sum_node_link_values = dict()

    for row in map_graph:
        ent = str(row[2]).lower()
        sum_node = str(row[0]).lower()
        sum2orgNodes_dict[sum_node].append(ent)
    write_to_csv(f'../data/{dataset_name}/mappings/{dataset_name}_mapped_sum2orgNode.{output_format}', sum2orgNodes_dict)

    for node in sum2orgNodes_dict:
        link_value = 1/len(sum2orgNodes_dict[node])
        sum_node_link_values[str(node).lower()]=link_value
 
    sum2type_dict = dict()
    org2type_dict = dfOrg2Type.to_dict(orient='index')

    for sumNode in sum2orgNodes_dict:
        map_dict = dict.fromkeys(labels, 0.0)
        for orgNode in sum2orgNodes_dict[sumNode]:
            if orgNode in org2type_dict.keys():
                mapping = org2type_dict[orgNode]
                for ent_type in mapping:
                    if mapping[ent_type]== float(1.0):
                        map_dict[ent_type] += sum_node_link_values[sumNode]
        sum2type_dict[sumNode] = map_dict

    #save to csv, need this later 
    dfSum2type = pd.DataFrame.from_dict(sum2type_dict, orient='index')
    dfSum2type.to_csv(f'../data/{dataset_name}/mappings/{dataset_name}_mapped_sum2type.{output_format}')

    return sum2orgNodes_dict, sum2type_dict, org2type_dict
   
def main_map_org_nodes():
    dataset = 'AIFB'
    labels = map_org_labels(dataset)
    dfOrg2type = create_map_csv('AIFB', labels)
    sum2orgNodes_dict, sum2type_dict, org2type_dict = map_sum_nodes(dataset, dfOrg2type, labels)

if __name__=='__main__':
    main_map_org_nodes()
