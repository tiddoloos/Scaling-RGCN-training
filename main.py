from model.modelTrainer import modelTrainer
from helpers.plot import plot_main

def initialize_training() -> None:
    """This functions executes the experiment to scale grpah training for multi class entity prediction.
    After training on multiple summary graphs graphs, 
    the weights of the summary model will be transferd to a new model for training on the original graph.
    Also a benchmark experiment is conducted.
    """

    hidden_l = 16
    epochs = 50
    weight_d = 0.0005
    lr = 0.01

    # Transfer learning expriment
    trainer = modelTrainer('AIFB', hidden_l)
    results_transfer = trainer.main_modelTrainer(epochs, weight_d, lr, benchmark=False)
   
    # Benchmark
    results_benchmark = trainer.main_modelTrainer(epochs, weight_d, lr, benchmark=True)

    results = {**results_transfer, **results_benchmark}

    plot_main('AIFB', results, epochs)


if __name__=='__main__':
    initialize_training()