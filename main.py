from model.modelTrainer import modelTrainer
from helpers.plot import main_plot
import argparse

parser = argparse.ArgumentParser(description='experiment arguments')
parser.add_argument('-dataset', type=str, choices=['AIFB', 'MUTAG', 'AM'], help='inidcate dataset name')
args = vars(parser.parse_args())


def initialize_training() -> None:
    """This functions executes the experiment to scale grpah training for multi class entity prediction.
    After training on summary graphs, 
    the weights of the summary model will be transferd to a new model for training on the original graph.
    Also a benchmark experiment is conducted.
    """
    
    hidden_l = 16
    epochs = 51
    weight_d = 0.0005
    lr = 0.01

    # Transfer learning expriment
    trainer = modelTrainer(args['dataset'], hidden_l)
    results_transfer = trainer.main_modelTrainer(epochs, weight_d, lr, exp='transfer')

    #only init with node ebedding layer and no weight transfer or embedding trick
    results_embedding = trainer.main_modelTrainer(epochs, weight_d, lr, exp='embedding')
   
    # Benchmark
    results_benchmark = trainer.main_modelTrainer(epochs, weight_d, lr, exp='benchmark')

    results = {**results_transfer, **results_embedding, **results_benchmark}

    main_plot(args['dataset'], results, epochs)


if __name__=='__main__':
    initialize_training()