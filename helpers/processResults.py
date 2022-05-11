import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime


def plot_and_save(metric, dataset, results_dict, epochs, exp):
        # save results as txt or json
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
        str_date = dt.strftime('%d-%B-%Y-%I:%M-%p')
        plt.savefig(f'./output_dir/AIFB/{dataset}_{metric}_{exp}_{str_date}.png')

        with open(f'./output_dir/AIFB/{dataset}_{metric}_{exp}_{str_date}.json', 'w') as write_file:
                json.dump(results_dict, write_file, indent=4)
