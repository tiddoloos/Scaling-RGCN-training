import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def main_plot(metric, dataset, results_dict, epochs, exp):
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
        # plt.savefig('')
        plt.show()

        dt = datetime.now()
        str_date = dt.strftime('%d%B-%Y')
        plt.savefig(f'"$TMPDIR"/output_dir/{metric}_{exp}_{str_date}.png')
