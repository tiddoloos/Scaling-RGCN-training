import argparse

from copy import deepcopy
from typing import Callable, Dict

from graphdata.dataset import Dataset
from helpers.processResults import plot_results, save_to_json, print_max_acc, get_av_results_dict
from helpers import timing
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.models import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from model.modelTrainer import Trainer


def initialize_expirements(args: Dict, experiments: Dict[str, Dict[str, Callable]]) -> None:
    """This functions executes experiments to scale graph training with RGCN. 
    After training on summary graphs, the weights of and node embeddings of 
    the summary model will be transferd to a new model for training on the 
    original graph. Also a baseline experiment is carried out.
    """
    hidden_l = 32
    lr = 0.01
    weight_d = 0.0005

    acc_dicts_list = []
    loss_dicts_list = []

    k = args['k']
    for j in range(k):

        # initialzie the data and use deepcopy to keep original data unchanged.
        data = Dataset(args['dataset'])
        data.init_dataset(args['emb'])

        results_exp_acc = dict()
        results_exp_loss = dict()
        
        if args['exp'] == None:
            for exp, exp_settings in experiments.items():
                trainer = Trainer(deepcopy(data), hidden_l, args['epochs'], args['emb'], lr, weight_d)
                results_acc, results_loss = trainer.exp_runner(exp_settings['sum_layers'], exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
                results_exp_acc.update(results_acc)
                results_exp_loss.update(results_loss)
                timing.log(f'{exp} experiment done')
        
        else:
            exp = args['exp']
            exp_settings = experiments[exp]
            trainer = Trainer(deepcopy(data), hidden_l, args['epochs'], args['emb'], lr, weight_d)
            results_acc, results_loss = trainer.exp_runner(exp_settings['sum_layers'], exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
            results_exp_acc.update(results_acc)
            results_exp_loss.update(results_loss)
            timing.log(f'{exp} experiment done')

            exp = 'baseline'
            exp_settings = experiments[exp]
            trainer = Trainer(deepcopy(data), hidden_l, args['epochs'], args['emb'], lr, weight_d)
            results_baseline_acc, results_baseline_loss = trainer.exp_runner(exp_settings['sum_layers'], exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
            results_exp_acc.update(results_baseline_acc)
            results_exp_loss.update(results_baseline_loss)
            timing.log(f'{exp} experiment done')

        acc_dicts_list.append(results_exp_acc)
        loss_dicts_list.append(results_exp_loss)

    av_acc_results = get_av_results_dict(k, acc_dicts_list)
    av_loss_results = get_av_results_dict(k, loss_dicts_list)
  
    print_max_acc(av_acc_results)

    save_to_json('avg_Accuracy', args['dataset'], args['exp'], k, av_acc_results)
    save_to_json('avg_Loss', args['dataset'], args['exp'], k, av_loss_results)

    plot_results('Avg accuracy', args['dataset'], args['exp'], args['epochs'], k, av_acc_results)
    plot_results('Avg loss', args['dataset'], args['exp'], args['epochs'], k, av_loss_results)


parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'MUTAG', 'AM', 'TEST'], help='inidcate dataset name', default='AIFB')
parser.add_argument('-exp', type=str, choices=['sum', 'mlp', 'attention', 'embedding'], help='select experiment')
parser.add_argument('-epochs', type=int, default=51, help='indicate number of training epochs')
parser.add_argument('-emb', type=int, default=63, help='indicate number of training epochs')
parser.add_argument('-k', type=int, default=1, help='indicate experiment iterations')
args = vars(parser.parse_args())

experiments = {
'sum': {'sum_layers': Emb_Layers, 'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
'mlp': {'sum_layers': Emb_Layers, 'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
'attention': {'sum_layers': Emb_Layers, 'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True},
'baseline': {'sum_layers': None, 'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False}
}


if __name__=='__main__':
    initialize_expirements(args, experiments)
