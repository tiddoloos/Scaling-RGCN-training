from traceback import print_list
import torch

from torch import embedding, nn
from torch import Tensor
from torch_geometric.data import Data
from typing import List, Tuple, Callable

from graphdata.graphData import Dataset, Graph
from model.embeddingTricks import init_embedding


class Trainer:
    device = torch.device(str('cuda:0') if torch.cuda.is_available() else 'cpu')
    def __init__(self, data: Dataset, hidden_l: int, epochs: int, emb_dim: int, lr: float, weight_d: float):
        self.data = data
        self.hidden_l = hidden_l
        self.epochs = epochs
        self.emb_dim = emb_dim
        self.lr = lr
        self.weight_d = weight_d

    def transfer_weights(self, sumModel, orgModel) -> None:
        # rgcn1 
        weight_sg_1 = sumModel.rgcn1.weight
        bias_sg_1 = sumModel.rgcn1.bias
        root_sg_1 = sumModel.rgcn1.root

        # rgcn2
        weight_sg_2 = sumModel.rgcn2.weight
        bias_sg_2 = sumModel.rgcn2.bias
        root_sg_2 = sumModel.rgcn2.root

        # transfer
        orgModel.override_params(weight_sg_1, bias_sg_1, root_sg_1, weight_sg_2, bias_sg_2, root_sg_2)
        print('weight transfer done')

    def calc_acc(self, pred: Tensor, x: Tensor, y: Tensor) -> float:
        tot = torch.sum(y == 1).item()
        p = (torch.sum((pred[x] == y) * (pred[x] == 1))) / tot
        return p.item()
    
    def evaluate(self, model: nn.Module, traininig_data: Data) -> float:
        pred = model(traininig_data)
        pred = torch.round(pred)
        acc = self.calc_acc(pred, self.data.orgGraph.training_data.x_val, self.data.orgGraph.training_data.y_val)
        return acc

    def train(self, model: nn.Module, graph: Graph, sum_graph=True) -> Tuple[List, List]:
        model = model.to(self.device)
        training_data = graph.training_data.to(self.device)
        loss_f = torch.nn.BCELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_d)

        accuracies = []
        losses = []

        for epoch in range(self.epochs):
            if not sum_graph:
                model.eval()
                acc = self.evaluate(model, training_data)
                accuracies.append(acc)
            model.train()
            optimizer.zero_grad()
            out = model(training_data)
            targets = training_data.y_train.to(torch.float32)
            output = loss_f(out[training_data.x_train], targets)
            output.backward()
            optimizer.step()
            l = output.item()
            losses.append(l)
            if not sum_graph:
                    print(f'Accuracy on validation set = {acc}')
            if epoch%10==0:
                print(f'Epoch: {epoch}, Loss: {l:.4f}')
        
        return accuracies, losses
    
    def exp_runner(self, sum_layers: nn.Module, org_layers: nn.Module, embedding_trick: Callable, transfer: bool, exp: str):
        print(3*'-', f'{exp.upper()} EXPERIMENT', 3*'-')
        results_acc = dict()
        results_loss = dict()

        if transfer == True:
            print('Training on Summary Graphs...')
            sumModel = sum_layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes, self.emb_dim)
            for count, sum_graph in enumerate(self.data.sumGraphs):
                _, results_loss[f'Sum Loss {count}'] = self.train(sumModel, sum_graph)

        orgModel = org_layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes, self.emb_dim)
    
        if embedding_trick != None:
            embedding_trick(self.data.orgGraph, self.data.sumGraphs, self.emb_dim)

        if transfer == True:
            self.transfer_weights(sumModel, orgModel)

        print('Training on Orginal Graph...')
        results_acc[f'{exp} Accuracy'], results_loss[f'{exp} Loss'] = self.train(orgModel, self.data.orgGraph, sum_graph=False)

        return results_acc, results_loss