import argparse
import itertools

from collections import defaultdict
from typing import Dict, List

from helpers.processResults import plot_results
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.models import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from main import initialize_expirements

"""use this file to run experiments in main.py multiple times en get an average result for accuracy and loss"""

def add_data(list1: List[int], list2: List[int]):
    temp_list= list(itertools.zip_longest(list1, list2, fillvalue = 0))
    return [x+y for x,y in temp_list]

def get_av_results_dict(k: int, dicts_list: List[Dict[str, int]]):
    av_results_dict = defaultdict(list)
    for key, int_list in dicts_list[0].items():
        for dict in dicts_list[1:]:
            av_results_dict[key] = add_data(int_list, dict[key])
        av_results_dict[key][:] = [x / k for x in av_results_dict[key]]
    return av_results_dict
    
def run_k_times(args: Dict[str, str], experiments):
    k = args['k']
    acc_dicts_list = []
    loss_dicts_list = []
    for i in range(k):
        acc_dict, loss_dict = initialize_expirements(args, experiments, plot=False, k_run=True)
        acc_dicts_list.append(acc_dict)
        loss_dicts_list.append(loss_dict)

    av_acc_results = get_av_results_dict(k, acc_dicts_list)
    av_loss_results = get_av_results_dict(k, loss_dicts_list)

    plot_results('Average Accuracy', args['dataset'], args['exp'], args['epochs'], av_acc_results)
    plot_results('Average Loss', args['dataset'], args['exp'], args['epochs'], av_loss_results)
    

experiments = {
'sum': {'sum_layers': Emb_Layers, 'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
'mlp': {'sum_layers': Emb_Layers, 'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
'attention': {'sum_layers': Emb_Layers, 'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True},
'baseline': {'sum_layers': None, 'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False}
}

parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'MUTAG', 'AM', 'TEST'], help='inidcate dataset name', default='AIFB')
parser.add_argument('-exp', type=str, choices=['sum', 'mlp', 'attention', 'embedding'], help='select experiment')
parser.add_argument('-epochs', type=int, default=51, help='indicate number of training epochs')
parser.add_argument('-emb', type=int, default=63, help='indicate number of training epochs')
parser.add_argument('-k', type=int, default=3, help='indicate experiment iterations')
args = vars(parser.parse_args())

results_path = './results/'
run_k_times(args, experiments)
