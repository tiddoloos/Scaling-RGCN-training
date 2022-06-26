import argparse

from copy import deepcopy
from typing import Callable, Dict, Union

from graphdata.dataset import Dataset
from graphdata.createAttributeSum import create_sum_map
from helpers.results import Results
from helpers import timing
from helpers.checks import do_checks
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.layers import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from model.modelTrainer import Trainer


def initialize_expirements(configs: Dict[str, Union[bool, str, int, float]], methods: Dict[str, Dict[str, Callable]], org_path: str, sum_path: str, map_path: str) -> None:
    """This functions executes experiments to scale graph training with RGCN. 
    After training on summary graphs, the weights and node embeddings of 
    the summary model will be transferd to a new model for training on the 
    original graph. Also a baseline experiment is carried out.
    """

    # before running program, do some check and assert or adjust settings if needed
    configs = do_checks(configs, sum_path, map_path)

    results = Results()

    for j in range(configs['i']):

        # create summaries
        if configs['create_attr_sum']:
            timing.log('Creating graph summaries...')
            create_sum_map(org_path, sum_path, map_path, dataset)

        # initialzie the data and use deepcopy when using data to keep original data unchanged.
        timing.log('...Making Graph data...')
        data = Dataset(org_path, sum_path, map_path)
        data.init_dataset()
        
        if configs['exp'] == None:
            trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)
            trainer.train_summaries()
            for exp, exp_settings in methods['experiments'].items():
                results.add_key(exp)
                timing.log(f'Start {exp} Experiment')
                results_acc, results_loss, results_f1_w, results_f1_m, test_acc, test_micro, test_macro = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], configs, exp)
                
                for result in [results_acc, results_loss, results_f1_w, results_f1_m]:
                    results.update_run_results(result, exp)

                results.test_accs[f'Test acc {exp}'].append(test_acc)
                results.test_f1_weighted[f'Test F1 weighted {exp}'].append(test_micro)
                results.test_f1_macro[f'Test F1 macro {exp}'].append(test_macro) 

                timing.log(f'{exp} experiment done')
        
        elif configs['exp'] != 'baseline':
                exp = configs['exp']
                results.add_key(exp)
                exp_settings = methods['experiments'][exp]
                trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)

                timing.log('Training on summary Graphs')
                trainer.train_summaries()

                timing.log(f'Start {exp} Experiment')
                results_acc, results_loss, results_f1_w, results_f1_m, test_acc, test_micro, test_macro = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], configs, exp)
                timing.log(f'{exp} experiment done')

                for result in [results_acc, results_loss, results_f1_w, results_f1_m]:
                    results.update_run_results(result, exp)

                results.test_accs[f'Test acc {exp}'].append(test_acc)
                results.test_f1_weighted[f'Test F1 weighted {exp}'].append(test_micro)
                results.test_f1_macro[f'Test F1 macro {exp}'].append(test_macro) 


        exp = 'baseline'
        exp_settings = methods[exp]
        results.add_key(exp)
        trainer = Trainer(deepcopy(data), configs['hl'], configs['epochs'], configs['emb'], configs['lr'], weight_d=0.00005)

        timing.log(f'Start {exp} Experiment')
        results_acc, results_loss, results_f1_w, results_f1_m, test_acc, test_micro, test_macro = trainer.train_original(exp_settings['org_layers'], exp_settings['embedding_trick'], configs, exp)
        timing.log(f'{exp} experiment done')

        for result in [results_acc, results_loss, results_f1_w, results_f1_m]:
            results.update_run_results(result, exp)

        results.test_accs[f'Test acc {exp}'].append(test_acc)
        results.test_f1_weighted[f'Test F1 weighted {exp}'].append(test_micro)
        results.test_f1_macro[f'Test F1 macro {exp}'].append(test_macro) 

    results.process_results(configs)


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
                    'org_layers': Emb_Layers, 'embedding_trick': None},
                'experiments': {
                    'summation': {'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings},
                    'mlp': {'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings},
                    'attention': {'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings}}}

    dataset = configs['dataset']
    sum = configs['sum']
    path = f'graphdata/{dataset}/{dataset}_complete.nt'
    sum_path = f'graphdata/{dataset}/{sum}/sum/'
    map_path = f'graphdata/{dataset}/{sum}/map/'

    initialize_expirements(configs, methods, path, sum_path, map_path)
