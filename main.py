import argparse
from collections import defaultdict

from copy import deepcopy
from typing import Callable, Dict

from graphdata.dataset import Dataset
from graphdata.createAttributeSum import create_sum_map
from helpers.processResults import plot_results, save_to_json, create_run_report, get_av_results_dict
from helpers import timing
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.layers import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from model.modelTrainer import Trainer



def initialize_expirements(configs: Dict, methods: Dict[str, Dict[str, Callable]], path: str, sum_path: str, map_path: str, graph_pros_test: bool = False) -> None:
    """This functions executes experiments to scale graph training with RGCN. 
    After training on summary graphs, the weights and node embeddings of 
    the summary model will be transferd to a new model for training on the 
    original graph. Also a baseline experiment is carried out.
    """

    acc_dicts_list = []
    loss_dicts_list = []
    test_acc_collect = defaultdict(list)

    iter = configs['i']
    for j in range(configs['i']):

        # create summaries
        if configs['sum'] == 'attr':
            timing.log('Creating graph summaries...')
            create_sum_map(path, sum_path, map_path)

        # initialzie the data and use deepcopy when using data to keep original data unchanged.
        timing.log('...Making Graph data...')
        data = Dataset(configs['dataset'], configs['sum'])
        data.init_dataset()

        if graph_pros_test:
            return

        results_exp_acc = dict()
        results_exp_loss = dict()


        if configs['exp'] == None:
            trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)
            trainer.train_summaries(methods['baseline']['org_layers'])
            for exp, exp_settings in methods['experiments'].items():
                timing.log(f'Start {exp} Experiment')
                results_acc, results_loss, test_acc = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
                results_exp_acc.update(results_acc)
                results_exp_loss.update(results_loss)
                test_acc_collect[f'Test set {exp}'].append(test_acc)
                timing.log(f'{exp} experiment done')
        
        else:
            if configs['exp'] != 'baseline':
                exp = configs['exp']
                exp_settings = methods['experiments'][exp]
                trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)

                timing.log('Training on summary Graphs')
                trainer.train_summaries(methods['baseline']['org_layers'])

                timing.log(f'Start {exp} Experiment')
                results_acc, results_loss, test_acc  = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
                timing.log(f'{exp} experiment done')

                results_exp_acc.update(results_acc)
                results_exp_loss.update(results_loss)
                test_acc_collect[f'Test set {exp}'].append(test_acc)

        exp = 'baseline'
        exp_settings = methods[exp]
        trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)

        timing.log(f'Start {exp} Experiment')
        results_baseline_acc, results_baseline_loss, test_acc = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
        timing.log(f'{exp} experiment done')

        results_exp_acc.update(results_baseline_acc)
        results_exp_loss.update(results_baseline_loss)
        test_acc_collect[f'Test set {exp}'].append(test_acc)

        acc_dicts_list.append(results_exp_acc)
        loss_dicts_list.append(results_exp_loss)


    # porocessing results
    av_acc_results = get_av_results_dict(iter, acc_dicts_list)
    av_loss_results = get_av_results_dict(iter, loss_dicts_list)

    create_run_report(f'{configs["sum"]}_report', configs, configs['dataset'], configs['exp'], iter, av_acc_results, test_acc_collect)

    save_to_json(f'{configs["sum"]}_avg_acc', configs['dataset'], configs['exp'], iter, av_acc_results)
    save_to_json(f'{configs["sum"]}_avg_loss', configs['dataset'], configs['exp'], iter, av_loss_results)
    
    plot_results(f'{configs["sum"]}_Accuracy', configs['dataset'], configs['exp'], configs['epochs'], iter, av_acc_results)
    plot_results(f'{configs["sum"]}_Loss', configs['dataset'], configs['exp'], configs['epochs'], iter, av_loss_results)




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='experiment arguments')
    parser.add_argument('-dataset', type=str, choices=['AIFB', 'AIFB1', 'BGS', 'MUTAG', 'AM', 'AM1', 'TEST'], help='inidcate dataset name', default='AIFB')
    parser.add_argument('-sum', type=str, choices=['attr', 'bisim', 'mix'], default='attr', help='summarization technique')
    parser.add_argument('-exp', type=str, choices=['summation', 'mlp', 'attention', 'baseline'], help='select experiment')
    parser.add_argument('-epochs', type=int, default=51, help='indicate number of training epochs')
    parser.add_argument('-emb', type=int, default=63, help='Node embediding dimension')
    parser.add_argument('-i', type=int, default=1, help='experiment iterations')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-hl', type=int, default=16, help='hidden layer size')
    configs = vars(parser.parse_args())

    methods = {'baseline': {
                    'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False},
            'experiments': {
                    'summation': {'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
                    'mlp': {'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
                    'attention': {'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True}}}

    dataset = configs['dataset']
    path = f'graphdata/{dataset}/{dataset}_complete.nt'
    sum_path = f'graphdata/{dataset}/attr/sum/{dataset}_sum_'
    map_path = f'graphdata/{dataset}/attr/map/{dataset}_map_'

    initialize_expirements(configs, methods, path, sum_path, map_path)
