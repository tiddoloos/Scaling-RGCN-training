import argparse
from model.modelTrainer import modelTrainer
from helpers.plot import main_plot

parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'MUTAG', 'AM'], help='inidcate dataset name')
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
    embedding_dimension = 63

    # Transfer learning expriment
    trainer = modelTrainer(args['dataset'], hidden_l)
    if args['exp'] == None:
        results_transfer = trainer.main_modelTrainer(epochs, weight_d, lr, embedding_dimension,  exp='sum')

        #train mlp to create embeddigs for original grpah
        results_mlp = trainer.main_modelTrainer(epochs, weight_d, lr, embedding_dimension,  exp='mlp')

        #train attention layer to create embeddigs for original grpah
        results_att = trainer.main_modelTrainer(epochs, weight_d, lr, embedding_dimension, exp='attention')

        #only init with node ebedding layer and no weight transfer or embedding trick
        results_embedding = trainer.main_modelTrainer(epochs, weight_d, lr, embedding_dimension,  exp='embedding')

        results_exp = {**results_att, **results_transfer, **results_embedding, **results_mlp}
    
    else:
        results_exp = trainer.main_modelTrainer(epochs, weight_d, lr, embedding_dimension,  exp=args['exp'])

    results_baseline = trainer.main_modelTrainer(epochs, weight_d, lr, embedding_dimension,  exp='baseline')


    results = {**results_exp, **results_baseline}

    main_plot(args['dataset'], results, epochs)


if __name__=='__main__':
    initialize_training()