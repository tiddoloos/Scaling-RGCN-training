import argparse
from collections import defaultdict

from copy import deepcopy
from typing import Callable, Dict

from graphdata.dataset import Dataset
from helpers.processResults import plot_results, save_to_json, create_run_report, get_av_results_dict
from helpers import timing
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.layers import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from model.modelTrainer import Trainer


def initialize_expirements(configs: Dict, methods: Dict[str, Dict[str, Callable]], graph_pros_test = False) -> None:
    """This functions executes experiments to scale graph training with RGCN. 
    After training on summary graphs, the weights and node embeddings of 
    the summary model will be transferd to a new model for training on the 
    original graph. Also a baseline experiment is carried out.
    """

    weight_d = 0.00005

    acc_dicts_list = []
    loss_dicts_list = []
    test_acc_collect = defaultdict(list)

    iter = configs['i']
    for j in range(configs['i']):

        # initialzie the data and use deepcopy to keep original data unchanged.
        data = Dataset(configs['dataset'])
        data.init_dataset()

        if graph_pros_test:
            return

        results_exp_acc = dict()
        results_exp_loss = dict()

        if configs['exp'] == None:
            trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d)
            trainer.train_summaries(methods['baseline']['org_layers'])
            for exp, exp_settings in methods['experiments'].items():
                results_acc, results_loss, test_acc = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
                results_exp_acc.update(results_acc)
                results_exp_loss.update(results_loss)
                test_acc_collect[f'Test set {exp}'].append(test_acc)
                timing.log(f'{exp} experiment done')
        
        else:
            if configs['exp'] != 'baseline':
                exp = configs['exp']
                exp_settings = methods['experiments'][exp]
                trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d)
                trainer.train_summaries(methods['baseline']['org_layers'])
                results_acc, results_loss, test_acc  = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
                results_exp_acc.update(results_acc)
                results_exp_loss.update(results_loss)
                test_acc_collect[f'Test set {exp}'].append(test_acc)
                timing.log(f'{exp} experiment done')

        exp = 'baseline'
        exp_settings = methods[exp]
        trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d)
        results_baseline_acc, results_baseline_loss, test_acc = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
        results_exp_acc.update(results_baseline_acc)
        results_exp_loss.update(results_baseline_loss)
        test_acc_collect[f'Test set {exp}'].append(test_acc)
        timing.log(f'{exp} experiment done')

        acc_dicts_list.append(results_exp_acc)
        loss_dicts_list.append(results_exp_loss)

    av_acc_results = get_av_results_dict(iter, acc_dicts_list)
    av_loss_results = get_av_results_dict(iter, loss_dicts_list)

    create_run_report('report', configs, configs['dataset'], configs['exp'], iter, av_acc_results, test_acc_collect)

    save_to_json('avg_acc', configs['dataset'], configs['exp'], iter, av_acc_results)
    save_to_json('avg_loss', configs['dataset'], configs['exp'], iter, av_loss_results)
    
    plot_results('Accuracy', configs['dataset'], configs['exp'], configs['epochs'], iter, av_acc_results)
    plot_results('Loss', configs['dataset'], configs['exp'], configs['epochs'], iter, av_loss_results)


parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'AIFB1', 'BGS', 'MUTAG', 'AM', 'AM1', 'TEST'], help='inidcate dataset name', default='AIFB')
parser.add_argument('-exp', type=str, choices=['summation', 'mlp', 'attention', 'baseline'], help='select experiment')
parser.add_argument('-epochs', type=int, default=51, help='indicate number of training epochs')
parser.add_argument('-emb', type=int, default=63, help='indicate number of training epochs')
parser.add_argument('-i', type=int, default=1, help='indicate experiment iterations')
parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-hl', type=int, default=16, help='hidden layer size')
configs = vars(parser.parse_args())

methods = {'baseline': {
                'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False},
            'experiments': {
                'summation': {'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
                'mlp': {'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
                'attention': {'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True}}}


if __name__=='__main__':
    initialize_expirements(configs, methods)
