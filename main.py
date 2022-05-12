import argparse
from model.modelTrainer import modelTrainer
from helpers.processResults import plot_and_save
from experiments import run_experiment
from helpers import timing

parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'MUTAG', 'AM', 'TEST'], help='inidcate dataset name')
parser.add_argument('-exp', type=str, choices=['sum', 'mlp', 'attention', 'embedding'], help='select experiment')
args = vars(parser.parse_args())


def initialize_training() -> None:
    """This functions executes the experiment to scale grpah training for multi class entity prediction.
    After training on summary graphs, 
    the weights of the summary model will be transferd to a new model for training on the original graph.
    Also a baseline experiment is conducted.
    """

    hidden_l = 16
    epochs = 51
    weight_d = 0.0005
    lr = 0.01
    #embedding dimension must be devisible by the number of heads aka number of graph summaries (attention layer)
    embedding_dimension = 63

    trainer = modelTrainer(args['dataset'], hidden_l)
    # Transfer learning expriment
    if args['exp'] == None:
        results_transfer_acc, results_transfer_loss = run_experiment(trainer, epochs, weight_d, lr, embedding_dimension,  exp='sum')
        timing.log('experiment done')

        #train mlp to create embeddigs for original graph
        # trainer = modelTrainer(args['dataset'], hidden_l)
        results_mlp_acc, results_mlp_loss = run_experiment(trainer, epochs, weight_d, lr, embedding_dimension,  exp='mlp')
        timing.log('experiment done')

        #train attention layer to create embeddigs for original grpah
        # trainer = modelTrainer(args['dataset'], hidden_l)
        results_att_acc, results_att_loss = run_experiment(trainer, epochs, weight_d, lr, embedding_dimension, exp='attention')
        timing.log('experiment done')

        #init with node ebedding layer and train on org grpah -> no weight transfer or embedding trick
        # trainer = modelTrainer(args['dataset'], hidden_l)
        results_embedding_acc, results_embedding_loss = run_experiment(trainer, epochs, weight_d, lr, embedding_dimension,  exp='embedding')
        timing.log('experiment done')

        results_exp_acc = {**results_att_acc, **results_transfer_acc, **results_embedding_acc, **results_mlp_acc}
        results_exp_loss = {**results_att_loss , **results_transfer_loss , **results_embedding_loss , **results_mlp_loss }
    
    else:
        # trainer = modelTrainer(args['dataset'], hidden_l)
        results_exp_acc, results_exp_loss = run_experiment(trainer, epochs, weight_d, lr, embedding_dimension,  exp=args['exp'])
        timing.log('experiment done')

    results_baseline_acc, results_baseline_loss = run_experiment(trainer, epochs, weight_d, lr, embedding_dimension,  exp='baseline')
    timing.log('experiment done')


    results_acc = {**results_exp_acc, **results_baseline_acc}
    results_loss = {**results_exp_loss, **results_baseline_loss}

    plot_and_save('Accuracy', args['dataset'], results_acc, epochs, args['exp'])
    plot_and_save('Loss', args['dataset'], results_loss, epochs, args['exp'])

if __name__=='__main__':
    initialize_training()