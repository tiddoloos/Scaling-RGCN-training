import matplotlib.pyplot as plt
import numpy as np


def main_plot(dataset, results_dict, epochs):
        epoch_list = [i for i in range(epochs)]
        for key, result in results_dict.items():
                y = result
                x = epoch_list 
                plt.plot(x, y, label = key)

        plt.title(f'Results on the {dataset} dataset')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy/Loss')
        plt.grid(color='b', linestyle='-', linewidth=0.1)
        plt.margins(x=0)
        plt.legend(loc='best')
        plt.xticks(np.arange(0, len(epoch_list), 10))
        plt.xlim(ymin=0)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylim(ymin=0)
        # plt.savefig('')
        plt.show()
