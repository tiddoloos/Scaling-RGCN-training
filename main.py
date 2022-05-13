import argparse

from typing import Callable, Dict
from torch import nn

# from experiments import run_experiment
from helpers.processResults import plot_and_save, print_max_result
from helpers import timing
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings, init_embedding
from model.models import emb_layers, emb_mlp_Layers, emb_att_Layers, base_Layers
from model.modelTrainer import Trainer



def initialize_expiremt(args: Dict[str, str], experiments: Dict[str, Callable]) -> None:
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

    results_exp_acc = dict()
    results_exp_loss = dict()

    # Initialize trainer here and create the data. Data is used and kept the same for each experiment for comparison
    trainer = Trainer(args['dataset'], hidden_l, epochs, embedding_dimension, lr, weight_d)
    
    if args['exp'] == None:
        for exp, exp_settings in experiments.items():
            results_acc, results_loss = trainer.exp_runner(exp_settings['sum_layers'], exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
            results_exp_acc.update(results_acc)
            results_exp_loss.update(results_loss)
            timing.log('experiment done')
    
    if args['exp'] != None:
        exp = args['exp']
        exp_settings = experiments[exp]
        results_exp_acc, results_exp_loss = trainer.exp_runner(exp_settings['sum_layers'], exp_settings['org_layers'], exp_settings['embedding_trick'], exp_settings['transfer'], exp)
        timing.log('experiment done')

    results_baseline_acc = dict()
    results_baseline_loss = dict()
    results_baseline_acc['baseline Accuracy'], results_baseline_loss['baseline Loss'] = trainer.train(trainer.baseModel, trainer.data.orgGraph, sum_graph=False)
    timing.log('experiment done')


    results_acc = {**results_exp_acc, **results_baseline_acc}
    results_loss = {**results_exp_loss, **results_baseline_loss}

    print_max_result(results_acc)

    plot_and_save('Accuracy', args['dataset'], results_acc, epochs, args['exp'])
    plot_and_save('Loss', args['dataset'], results_loss, epochs, args['exp'])


parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'MUTAG', 'AM', 'TEST'], help='inidcate dataset name')
parser.add_argument('-exp', type=str, choices=['sum', 'mlp', 'attention', 'embedding'], help='select experiment')
args = vars(parser.parse_args())

experiments = {
'sum': {'sum_layers': emb_layers, 'org_layers': emb_layers, 'embedding_trick': sum_embeddings, 'transfer': True},
'mlp': {'sum_layers': emb_layers, 'org_layers': emb_mlp_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
'attention': {'sum_layers': emb_layers, 'org_layers': emb_att_Layers, 'embedding_trick': stack_embeddings, 'transfer': True},
'embedding': {'sum_layers': None, 'org_layers': emb_layers, 'embedding_trick': None, 'transfer': False}}


if __name__=='__main__':
    initialize_expiremt(args, experiments)