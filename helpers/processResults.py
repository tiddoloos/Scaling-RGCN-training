import json
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from datetime import datetime
from typing import Dict, List
from torch import nn


def print_max_acc(metric: str, dataset: str, exp: str, k: int, results_dict: Dict[str, List[int]]) -> None:
    max_results = defaultdict(dict)
    for experiment, results in results_dict.items():
        exp_strip = experiment.replace(' Accuracy', '')
        max_acc = max(results)
        epoch = int(results.index(max_acc)) - 1 
        max_acc = max_acc*100
        print(f'{exp_strip.upper()}: After epoch {epoch}, Max accuracy {round(max_acc, 2)}%')
        max_results[experiment] = {'epoch': epoch, 'acc': max_acc}
    
    dt = datetime.now()
    str_date = dt.strftime('%d%B%Y-%H%M%S')
    with open(f'./results/{dataset}_{metric}_k={k}_{exp}_{str_date}.json', 'w') as write_file:
            json.dump(max_results, write_file, indent=4)

def plot_results(metric: str, dataset: str, exp: str, epochs: int, k: int,  results_dict: Dict[str, List[int]]):
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
    dt = datetime.now()
    str_date = dt.strftime('%d%B%Y-%H%M%S')
    plt.savefig(f'./results/{dataset}_{metric}_k={k}_{exp}_{str_date}.png', format='png')
    plt.show()

def save_to_json(metric: str, dataset: str, exp: str, k: int, results_dict: Dict[str, List[int]]) -> None:
    dt = datetime.now()
    str_date = dt.strftime('%d%B%Y-%H%M%S')
    with open(f'./results/{dataset}_{metric}_k={k}_{exp}_{str_date}.json', 'w') as write_file:
            json.dump(results_dict, write_file, indent=4)

def print_trainable_parameters(model: nn.Module, exp: str) -> int:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of trainable parameters for {exp.upper()} model: {trainable_params}')
    return trainable_params

def add_data(list1: List[int], list2: List[int]):
    temp_list = list(zip(list1, list2))
    return [x+y for x,y in temp_list]

def get_av_results_dict(k: int, dicts_list: List[Dict[str, int]]):
    av_results_dict = defaultdict(list)
    for key in dicts_list[0].keys():
        new_lst = [0 for i in range(0, len(dicts_list[0][key]))]
        for dict in dicts_list:
            new_lst = add_data(dict[key], new_lst)
        av_results_dict[key] = [x / k for x in new_lst]
    return av_results_dict