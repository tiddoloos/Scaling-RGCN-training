import json
import matplotlib.pyplot as plt
import numpy as np
import os

from collections import defaultdict
from datetime import datetime
from typing import Dict, Union
from torch import nn

from model.modelTrainer import Trainer


class Results:
    def __init__(self) -> None:
        self.run_results = dict()
        self.test_accs = defaultdict(list)
        self.test_f1_weighted = defaultdict(list)
        self.test_f1_macro = defaultdict(list)

    def add_key(self, key):
        if key  not in self.run_results.keys():
            self.run_results[key]=defaultdict(list)

    def update_run_results(self, new_results, exp):
        for key, value in new_results.items():
            self.run_results[exp][key].append(np.array(value))

    def print_trainable_parameters(self, model: nn.Module, exp: str, trainer: Trainer) -> int:
        """calculate and print trainable parameters of the models"""

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if exp != 'baseline':
            for sumGraph in trainer.data.sumGraphs:
                trainable_params += sumGraph.embedding.shape[0] * sumGraph.embedding.shape[1]
        print(f'number of trainable parameters for {exp.upper()} model: {trainable_params}')
        return trainable_params

    def make_av_run_results(self) -> None:
        for exp, value in self.run_results.items():
            for metric, array_list in value.items():
                mean_arr = np.mean(np.array(array_list), axis=0)
                mean_list = list(np.around(mean_arr, 4))
                mean_low = list(np.around(mean_arr - np.std(np.array(array_list), axis=0), 4))
                mean_up = list(np.around(mean_arr + np.std(np.array(array_list), axis=0), 4))
                self.run_results[exp][metric] = [mean_list, mean_low, mean_up]

    def create_run_report(self, path: str, configs: Dict[str, Union[str, int]]) -> None:
        "with this function we save a statistics report of the experiments"

        report = defaultdict(dict)
        report.update(configs)

        for experiment, metric_retsults in self.run_results.items():
            for metric, results in metric_retsults.items():
                max_metric = max(results[0])
                epoch = int(results[0].index(max_metric)) - 1 
                percentage_max = max_metric *100
                report[experiment][metric] = {'epoch': epoch, 'max': round(percentage_max, 2)}
        
        for test_dict in [self.test_accs, self.test_f1_weighted, self.test_f1_macro]:
            for experiment, results in test_dict.items():
                avg  = round(float((sum(results)/len(results))*100), 2)
                std = round(float(np.std((np.array(results)*100))), 2)
                report[experiment] = {'mean': avg, 'std': std}

        with open(f'{path}/report_{configs["exp"]}_{configs["sum"]}_i={configs["i"]}.json', 'w') as write_file:
                json.dump(report, write_file, indent=4)

    def plot_results(self, path: str, configs: Dict[str, Union[str, int]]):
        epoch_list = [j for j in range(configs['epochs'])]
        colors: dict = {'attention': '#FF0000', 'summation': '#069AF3', 'mlp': '#15B01A'}
        exps = self.run_results.keys()
        
        if configs['exp'] != 'baseline':
            with open(f'./baselines/{configs["dataset"]}_baseline/run_results_baseline_i=5.json') as baseline_results_file:
                b_results  = json.load(baseline_results_file)
        else:
            b_results = self.run_results
            for metric, result in b_results['baseline'].items():
                y_base = result[0][:51]
                y1_base = result[1][:51]
                y2_base = result[2][:51]
                x = epoch_list 
                ylim = 1.1
                step = 0.1

                for exp in exps:
                    if exp != 'baseline':                            
                        y = self.run_results[exp][metric][0]
                        y1 = self.run_results[exp][metric][1]
                        y2 = self.run_results[exp][metric][2]
                        x = epoch_list 
                        plt.fill_between(x, y1, y2, color=colors[exp], interpolate=True, alpha=0.2)
                        plt.plot(x, y, color=colors[exp], label=f'{exp} {metric}')

                    plt.fill_between(x, y1_base, y2_base, color='#FAC205', interpolate=True, alpha=0.45)
                    plt.plot(x, y_base, color='#FAC205', label = f'baseline {metric}')    
                    plt.title(f'{exp} {metric} on {configs["dataset"]} dataset during training epochs ({configs["sum"]})')
                    plt.xlabel('Epochs')
                    plt.ylabel(f'{metric}')
                    plt.grid(color='b', linestyle='-', linewidth=0.1)
                    plt.margins(x=0)
                    plt.legend(loc='best')
                    plt.xticks(np.arange(0, len(epoch_list), 5))
                    plt.xlim(xmin=0)

                    if max(y2_base) > 1:
                        ylim = round(max(y2_base)+1.0)
                        step = 0.5
                
                    plt.yticks(np.arange(0, ylim, step))
                    plt.ylim(ymin=0)
                    plt.savefig(f'{path}/{configs["dataset"]}_{exp}_{metric}_{configs["sum"]}_i={configs["i"]}.pdf', format='pdf')
                    plt.show()
                    plt.close()
                    plt.clf()
           
    def save_to_json(self, path: str, configs) -> None:
        with open(f'{path}/run_results_{configs["exp"]}_{configs["sum"]}_i={configs["i"]}.json', 'w') as write_file:
                json.dump(self.run_results, write_file, indent=4)

    def process_results(self, configs: Dict[str, Union[str, int]]) -> None:
        dt = datetime.now()
        str_date = dt.strftime('%d%B%Y-%H%M')
        path=f'./results/{configs["dataset"]}_{configs["exp"]}_{configs["sum"]}_i={configs["i"]}_{str_date}'
        os.mkdir(path)

        self.make_av_run_results()
        self.save_to_json(path, configs)
        self.create_run_report(path, configs)
        self.plot_results(path, configs)

        