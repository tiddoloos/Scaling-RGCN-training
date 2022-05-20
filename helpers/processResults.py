import json
import matplotlib.pyplot 
import numpy as np

from datetime import datetime
from typing import Dict, List
from torch import nn

def print_max_acc(results_dict: Dict[str, List[float]]) -> None:
        for exp, results in results_dict.items():
                exp_strip = exp.replace(' Accuracy', '')
                max_acc = max(results)
                epoch = int(results.index(max_acc)) - 1 
                max_acc = max_acc*100
                print(f'{exp_strip.upper()}: After epoch {epoch}, Max accuracy {round(max_acc, 2)}%')

def plot_results(metric: str, dataset: str, exp: str, epochs: int,  results_dict: Dict[str, List[int]]):
        plt = matplotlib.pyplot
        epoch_list = [i for i in range(epochs)]
        for key, result in results_dict.items():
                y = result
                x = epoch_list 
                plt.plot(x, y, label = key)

        plt.title(f'{metric} on {dataset} dataset during training epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy/Loss')
        plt.grid(color='b', linestyle='-', linewidth=0.1)
        plt.margins(x=0)
        plt.legend(loc='best')
        plt.xticks(np.arange(0, len(epoch_list), 5))
        plt.xlim(xmin=0)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylim(ymin=0)
        plt.show()
        dt = datetime.now()
        str_date = dt.strftime('%d%B%Y-%H%M%S')
        plt.savefig(f'./results/{dataset}_{metric}_{exp}_{str_date}.png', format='png')

def save_to_json(metric: str, dataset: str, exp: str, results_dict: Dict[str, List[int]]) -> None:
        dt = datetime.now()
        str_date = dt.strftime('%d%B%Y-%H%M%S')
        with open(f'./results/{dataset}_{metric}_{exp}_{str_date}.json', 'w') as write_file:
                json.dump(results_dict, write_file, indent=4)

def print_trainable_parameters(model: nn.Module, exp: str) -> int:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of trainable parameters for {exp.upper()} model: {trainable_params}')
    return trainable_params
