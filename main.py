import argparse
from collections import defaultdict

from copy import deepcopy
from typing import Callable, Dict

from graphdata.dataset import Dataset
from graphdata.createAttributeSum import create_sum_map
from helpers.processResults import process_results
from helpers import timing
from helpers.checks import do_checks
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.layers import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from model.modelTrainer import Trainer



def initialize_expirements(configs: Dict, methods: Dict[str, Dict[str, Callable]], org_path: str, sum_path: str, map_path: str) -> None:
    """This functions executes experiments to scale graph training with RGCN. 
    After training on summary graphs, the weights and node embeddings of 
    the summary model will be transferd to a new model for training on the 
    original graph. Also a baseline experiment is carried out.
    """

    # before running program, do some check and assert or adjust settings if needed
    configs = do_checks(configs, sum_path, map_path)

    acc_dicts_list = []
    loss_dicts_list = []
    f1_w_dicts_list = []
    f1_m_dicts_list = []
    test_accs = defaultdict(list)
    test_f1_weighted = defaultdict(list)
    test_f1_macro = defaultdict(list)

    for j in range(configs['i']):

        # create summaries
        if configs['create_attr_sum']:
            timing.log('Creating graph summaries...')
            create_sum_map(org_path, sum_path, map_path, dataset)

        # initialzie the data and use deepcopy when using data to keep original data unchanged.
        timing.log('...Making Graph data...')
        data = Dataset(org_path, sum_path, map_path)
        data.init_dataset()

        results_exp_acc = dict()
        results_exp_loss = dict()
        results_exp_f1_w = dict()
        results_exp_f1_m = dict()

        if configs['exp'] == None:
            trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)
            trainer.train_summaries(methods['baseline']['org_layers'])
            for exp, exp_settings in methods['experiments'].items():
                timing.log(f'Start {exp} Experiment')
                results_acc, results_loss, results_f1_w, results_f1_m, test_acc, test_micro, test_macro = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)

                results_exp_acc.update(results_acc)
                results_exp_loss.update(results_loss)
                results_exp_f1_w.update(results_f1_w)
                results_exp_f1_m.update(results_f1_m)

                test_accs[f'Test acc {exp}'].append(test_acc)
                test_f1_weighted[f'Test F1 weighted {exp}'].append(test_micro)
                test_f1_macro[f'Test F1 macro {exp}'].append(test_macro) 

                timing.log(f'{exp} experiment done')
        
        elif configs['exp'] != 'baseline':
                exp = configs['exp']
                exp_settings = methods['experiments'][exp]
                trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)

                timing.log('Training on summary Graphs')
                trainer.train_summaries(methods['baseline']['org_layers'])

                timing.log(f'Start {exp} Experiment')
                results_acc, results_loss, results_f1_w, results_f1_m, test_acc, test_micro, test_macro = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
                timing.log(f'{exp} experiment done')

                results_exp_acc.update(results_acc)
                results_exp_loss.update(results_loss)
                results_exp_f1_w.update(results_f1_w)
                results_exp_f1_m.update(results_f1_m)

                test_accs[f'Test acc {exp}'].append(test_acc)
                test_f1_weighted[f'Test F1 weighted {exp}'].append(test_micro)
                test_f1_macro[f'Test F1 macro {exp}'].append(test_macro) 

        exp = 'baseline'
        exp_settings = methods[exp]
        trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)

        timing.log(f'Start {exp} Experiment')
        results_baseline_acc, results_baseline_loss, results_baseline_f1_w, results_baseline_f1_m, test_acc, test_micro, test_macro = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
        timing.log(f'{exp} experiment done')

        results_exp_acc.update(results_baseline_acc)
        results_exp_loss.update(results_baseline_loss)
        results_exp_f1_w.update(results_baseline_f1_w)
        results_exp_f1_m.update(results_baseline_f1_m)

        test_accs[f'Test acc {exp}'].append(test_acc)
        test_f1_weighted[f'Test F1 weighted {exp}'].append(test_micro) 
        test_f1_macro[f'Test F1 macro {exp}'].append(test_macro) 

        acc_dicts_list.append(results_exp_acc)
        loss_dicts_list.append(results_exp_loss)
        f1_w_dicts_list.append(results_exp_f1_w)
        f1_m_dicts_list.append(results_exp_f1_m)

    # porocessing results
    process_results(configs, acc_dicts_list, loss_dicts_list, f1_w_dicts_list, f1_m_dicts_list, test_accs, test_f1_weighted, test_f1_macro)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='experiment arguments')
    parser.add_argument('-dataset', type=str, choices=['AIFB', 'AIFB1', 'BGS', 'MUTAG', 'AM', 'AM1', 'TEST2', 'TEST'], help='inidcate dataset name', default='AIFB')
    parser.add_argument('-sum', type=str, choices=['attr', 'bisim', 'mix'], default='attr', help='summarization technique')
    parser.add_argument('-exp', type=str, choices=['summation', 'mlp', 'attention', 'baseline'], help='select experiment')
    parser.add_argument('-epochs', type=int, default=51, help='indicate number of training epochs')
    parser.add_argument('-emb', type=int, default=63, help='Node embediding dimension')
    parser.add_argument('-i', type=int, default=1, help='experiment iterations')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-hl', type=int, default=16, help='hidden layer size')
    parser.add_argument('-e_trans', type=bool, default=True, help='emebdding transfer True/False')
    parser.add_argument('-w_trans', type=bool, default=True, help='RGCN weight transfer True/False')
    parser.add_argument('-create_attr_sum', type=bool, default=False, help='create attribute summaries before conducting the experiments')
    
    configs = vars(parser.parse_args())

    methods = {'baseline': {
                    'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False},
            'experiments': {
                    'summation': {'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
                    'mlp': {'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
                    'attention': {'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True}}}

    dataset = configs['dataset']
    sum = configs['sum']
    path = f'graphdata/{dataset}/{dataset}_complete.nt'
    sum_path = f'graphdata/{dataset}/{sum}/sum/'
    map_path = f'graphdata/{dataset}/{sum}/map/'

    initialize_expirements(configs, methods, path, sum_path, map_path)
