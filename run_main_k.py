import argparse

from collections import defaultdict
from typing import Dict, List

from helpers.processResults import plot_results, save_to_json
from main import initialize_expirements
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.models import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers



"""use this file to run experiments in main.py multiple times and get an average result for accuracy and loss"""

def add_data(list1: List[int], list2: List[int]):
    temp_list = list(zip(list1, list2))
    return [x+y for x,y in temp_list]

def get_av_results_dict(k: int, dicts_list: List[Dict[str, int]]):
    av_results_dict = defaultdict(list)

    for key in dicts_list[0].keys():
        new_lst = [0 for i in range(0, len(dicts_list[0][key]))]
        for dict in dicts_list:
            new_lst = add_data(dict[key], new_lst)
        av_results_dict[key] = [x / k for x in new_lst]
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

    save_to_json('avg_Accuracy', args['dataset'], args['exp'], av_acc_results)
    save_to_json('avg_Loss', args['dataset'], args['exp'], av_loss_results)

    plot_results('Avg Accuracy', args['dataset'], args['exp'], args['epochs'], av_acc_results)
    plot_results('Avg Average Loss', args['dataset'], args['exp'], args['epochs'], av_loss_results)
    

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

if __name__=='__main__':
    run_k_times(args, experiments)
