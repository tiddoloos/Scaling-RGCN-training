from model.modelTrainer import modelTrainer
from helpers.plot import plot_main

def initialize_training() -> None:
    epochs = 10
    weight_d = 0.0005
    lr = 0.01
    hidden_l=16

    # Transfer learning expriment
    trainer = modelTrainer('AIFB', hidden_l)
    results_transfer = trainer.main_modelTrainer(epochs, weight_d, lr, benchmark=False)
    # results_dict['Summary graph loss'], results_dict['Orginal graph accuracy'], results_dict['Orginal graph loss'] = trainer_trans.main_modelTrainer(epochs, weight_d, lr, benchmark=False)

    # Benchmark
    results_benchmark = trainer.main_modelTrainer(epochs, weight_d, lr, benchmark=True)
    results = {**results_transfer, **results_benchmark}

    plot_main('AIFB', results, epochs)


if __name__=='__main__':
    initialize_training()