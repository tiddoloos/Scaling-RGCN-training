import json
import matplotlib.pyplot
import numpy as np
import os

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Union
from torch import nn

def save_to_json(path: str, name: str, configs: Dict[str, Union[str, int]], results_dict: Dict[str, List[float]]) -> None:
    with open(f'{path}/{configs["dataset"]}_{name}_{configs["exp"]}_{configs["sum"]}_i={configs["i"]}.json', 'w') as write_file:
            json.dump(results_dict, write_file, indent=4)

def print_trainable_parameters(model: nn.Module, exp: str) -> int:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of trainable parameters for {exp.upper()} model: {trainable_params}')
    return trainable_params

def add_data(list1: List[int], list2: List[int]):
    temp_list = list(zip(list1, list2))
    return [x+y for x,y in temp_list]

def get_av_results_dict(i: int, dicts_list: List[Dict[str, int]]) -> Dict[str, List[List]]:
    av_results_dict = defaultdict(list)
    for key in dicts_list[0].keys():
        list_with_lists = [[] for i in range(len(dicts_list[0][key]))]
        for dict in dicts_list:
            for i, flt in enumerate(dict[key]):
                list_with_lists[i].append(flt)
        array = np.array(list_with_lists)
        av_results_dict[key].append(list(np.mean(array, axis=1)))
        av_results_dict[key].append(list(np.mean(array, axis=1) - np.std(array, axis=1)))
        av_results_dict[key].append(list(np.mean(array, axis=1) + np.std(array, axis=1)))
    return av_results_dict

def create_run_report(path: str, 
                        configs: Dict[str, Union[str, int]], 
                        results_dict: Dict[str, List[float]],  
                        test_accs: Dict[str, List[float]],
                        test_f1_micro: Dict[str, List[float]],
                        test_f1_macro: Dict[str, List[float]]
                        ) -> None:
    "with this function we save and print important statsitics of the experiment(s)"

    results_collection = defaultdict(dict)
    results_collection.update(configs)
    for experiment, results in results_dict.items():
        exp_strip = experiment.replace(' Accuracy', '')
        max_acc = max(results[0])
        epoch = int(results[0].index(max_acc)) - 1 
        max_acc = max_acc*100
        print(f'{exp_strip.upper()}: After epoch {epoch}, Max accuracy {round(max_acc, 2)}%')
        results_collection[experiment] = {'epoch': epoch, 'acc': max_acc}
    
    for test_dict in [test_accs, test_f1_micro, test_f1_macro]:
        for experiment, results in test_dict.items():
            avg  = float(sum(results)/len(results))
            std = float(np.std(np.array(results)))
            results_collection[experiment] = {'mean': avg, 'std': std}

    with open(f'{path}/{configs["dataset"]}_report_{configs["exp"]}_{configs["sum"]}_i={configs["i"]}.json', 'w') as write_file:
            json.dump(results_collection, write_file, indent=4)

def plot_results(path: str, stat: str, configs: Dict[str, Union[str, int]],  results_dict: Dict[str, List[float]]):
    epoch_list = [j for j in range(configs['epochs'])]
    
    keys = list(results_dict.keys())
    for key2 in keys:
            if key2.split(' ')[0] == 'baseline':
                result = results_dict[key2]
                y_base = result[0]
                y1_base = result[1]
                y2_base = result[2]
                x = epoch_list 
      
    for key1 in keys:
        if key1.split(' ')[0] != 'baseline':
            plt = matplotlib.pyplot
            result = results_dict[key1]
            y = result[0]
            y1 = result[1]
            y2 = result[2]
            x = epoch_list 

            plt.fill_between(x, y1, y2, interpolate=True, alpha=0.35)
            plt.plot(x, y, label = key1)

            plt.fill_between(x, y1_base, y2_base, interpolate=True, alpha=0.35)
            plt.plot(x, y_base, label = key2)
    
            plt.title(f'{key1} on {configs["dataset"]} dataset during training epochs ({configs["sum"]})')
            plt.xlabel('Epochs')
            plt.ylabel(f'{stat}')
            plt.grid(color='b', linestyle='-', linewidth=0.1)
            plt.margins(x=0)
            plt.legend(loc='best')
            plt.xticks(np.arange(0, len(epoch_list), 5))
            plt.xlim(xmin=0)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.ylim(ymin=0)
            plt.savefig(f'{path}/{configs["dataset"]}_{key1}_{configs["sum"]}_i={configs["i"]}.pdf', format='pdf')
            plt.show()

def process_results(configs: Dict[str, Union[str, int]], acc_dicts_list: List[Dict[str, float]], 
                    loss_dicts_list: List[Dict[str, float]], test_accs: Dict[str, List[float]], 
                    test_f1_micro: Dict[str, List[float]], test_f1_macro: Dict[str, List[float]]) -> None:
    
    dt = datetime.now()
    str_date = dt.strftime('%d%B%Y-%H%M')
    path=f'./results/{configs["dataset"]}_{configs["exp"]}_{configs["sum"]}_i={configs["i"]}_{str_date}'
    os.mkdir(path)
    av_acc_results = get_av_results_dict(configs["i"], acc_dicts_list)
    av_loss_results = get_av_results_dict(configs["i"], loss_dicts_list)

    create_run_report(path, configs, av_acc_results, test_accs, test_f1_micro, test_f1_macro)

    save_to_json(path, 'accuracy', configs, av_acc_results)
    save_to_json(path, 'loss', configs, av_loss_results)
    
    plot_results(path, 'Accuracy', configs, av_acc_results)
    plot_results(path, 'Loss', configs, av_loss_results)
    