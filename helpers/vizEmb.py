import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

from sklearn.manifold import TSNE


def viz_embedding(x: np.array, y: np.array, z: np.array, dataset: str, sum: str) -> None:
    sum_type = {'attr': 'Attribute', 'bisim': '(k)-f. bisim.'}
    plt.scatter(x, y, c=z, cmap='viridis_r', s=0.8)
    plt.title(f't-SNE transformed entity embedding ({dataset} {sum_type[sum]} summaries)')
    plt.savefig(f'./results/embeddings/{dataset}_{sum}_embedding.pdf', format='pdf')
    plt.show()
    plt.close()
    plt.clf()

def main_viz_emb(dataset: str, sum: str) -> None:
    embedding = torch.load(f'./results/embeddings/{dataset}_{sum}_embedding.pt')
    trans_emb = TSNE(init='pca').fit_transform(embedding)
    trans_emb_x, trans_emb_y = zip(*trans_emb)
    x, y = np.array(trans_emb_x), np.array(trans_emb_y)
    z = np.subtract(x, y)
    viz_embedding(x, y, z, dataset, sum)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='experiment arguments')
    parser.add_argument('-dataset', type=str, choices=['AIFB', 'BGS', 'MUTAG', 'AM', 'TEST'], help='inidcate dataset name', default='AIFB')
    parser.add_argument('-sum', type=str, choices=['attr', 'bisim'], help='summation type', default='attr')

    configs = vars(parser.parse_args())
    dataset = configs['dataset']
    sum = configs['sum']
    main_viz_emb(dataset, sum)
