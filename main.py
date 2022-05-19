import argparse

from copy import deepcopy
from typing import Callable, Dict

from graphdata.dataset import Dataset
from helpers.processResults import plot_results, save_to_json, print_max_result
from helpers import timing
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.models import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from model.modelTrainer import Trainer


def initialize_expirements(args: Dict[str, str], experiments: Dict[str, Dict[str, Callable]], plot=True) -> None:
    """This functions executes experiments to scale graph training with RGCN. 
    After training on summary graphs, the weights of and node embeddings of 
    the summary model will be transferd to a new model for training on the 
    original graph. Also a baseline experiment is carried out.
    """

    hidden_l = 16
    epochs = 51
    lr = 0.01
    weight_d = 0.0005
    #embedding dimension must be devisible by the number of summary graphs (attention layer)
    embedding_dimension = 63

    # initialzie the data and use deepcopy to keep original data unchanged.
    data = Dataset(args['dataset'])
    data.init_dataset(embedding_dimension)

    results_exp_acc = dict()
    results_exp_loss = dict()
    
    if args['exp'] == None:
        for exp, exp_settings in experiments.items():
            trainer = Trainer(deepcopy(data), hidden_l, epochs, embedding_dimension, lr, weight_d)
            results_acc, results_loss = trainer.exp_runner(exp_settings['sum_layers'], exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
            results_exp_acc.update(results_acc)
            results_exp_loss.update(results_loss)
            timing.log(f'{exp} experiment done')
    
    else:
        exp = args['exp']
        exp_settings = experiments[exp]
        trainer = Trainer(deepcopy(data), hidden_l, epochs, embedding_dimension, lr, weight_d)
        results_acc, results_loss = trainer.exp_runner(exp_settings['sum_layers'], exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
        results_exp_acc.update(results_acc)
        results_exp_loss.update(results_loss)
        timing.log(f'{exp} experiment done')

        exp = 'baseline'
        exp_settings = experiments[exp]
        trainer = Trainer(deepcopy(data), hidden_l, epochs, embedding_dimension, lr, weight_d)
        results_baseline_acc, results_baseline_loss = trainer.exp_runner(exp_settings['sum_layers'], exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
        results_exp_acc.update(results_baseline_acc)
        results_exp_loss.update(results_baseline_loss)
        timing.log(f'{exp} experiment done')

    print_max_result(results_exp_acc)

    save_to_json('Accuracy', args['dataset'], args['exp'], results_exp_acc)
    save_to_json('Accuracy', args['dataset'], args['exp'], results_exp_loss)


    if plot:
        plot_results('Accuracy', args['dataset'], args['exp'], epochs, results_exp_acc)
        plot_results('Accuracy', args['dataset'], args['exp'], epochs, results_exp_loss)


parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'MUTAG', 'AM', 'TEST'], help='inidcate dataset name')
parser.add_argument('-exp', type=str, choices=['sum', 'mlp', 'attention', 'embedding'], help='select experiment')
args = vars(parser.parse_args())

experiments = {
'sum': {'sum_layers': Emb_Layers, 'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
'mlp': {'sum_layers': Emb_Layers, 'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
'attention': {'sum_layers': Emb_Layers, 'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True},
'baseline': {'sum_layers': None, 'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False}
}


if __name__=='__main__':
    initialize_expirements(args, experiments)
