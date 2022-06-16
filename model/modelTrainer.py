import torch

from collections import defaultdict
from torch import nn
from torch import Tensor
from torch_geometric.data import Data
from typing import List, Tuple, Callable

from graphdata.graph import Graph
from graphdata.dataset import Dataset
from model.embeddingTricks import stack_embeddings


class Trainer:
    device = torch.device(str('cuda:0') if torch.cuda.is_available() else 'cpu')
    def __init__(self, data: Dataset, hidden_l: int, epochs: int, emb_dim: int, lr: float, weight_d: float):
        self.data: Dataset = data
        self.hidden_l: int = hidden_l
        self.epochs: int = epochs
        self.emb_dim: int = emb_dim
        self.lr: float = lr
        self.weight_d: float = weight_d
        self.sumModel: nn.Module = None

    def transfer_weights(self, orgModel) -> None:
        # rgcn1 
        weight_sg_1 = self.sumModel.rgcn1.weight.clone()
        bias_sg_1 = self.sumModel.rgcn1.bias.clone()
        root_sg_1 = self.sumModel.rgcn1.root.clone()

        # rgcn2
        weight_sg_2 = self.sumModel.rgcn2.weight.clone()
        bias_sg_2 = self.sumModel.rgcn2.bias.clone()
        root_sg_2 = self.sumModel.rgcn2.root.clone()

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
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_d)
        loss_f = torch.nn.BCELoss().to(self.device)

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
        
        # make space on GPU
        if self.device == 'cuda:0':
            training_data.to('cuda')
            model.to('cuda')
        return accuracies, losses

    def train_summaries(self, sum_layers: nn.Module):
        self.sumModel = sum_layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes, self.data.sumGraphs[0].num_nodes, self.emb_dim, len(self.data.sumGraphs))
        for _, sumGraph in enumerate(self.data.sumGraphs):
            self.sumModel.reset_embedding(sumGraph.num_nodes, self.emb_dim)
            _, _ = self.train(self.sumModel, sumGraph)
            sumGraph.training_data.embedding = self.sumModel.embedding.weight.clone() # or detach?

    def train_original(self, org_layers: nn.Module, embedding_trick: Callable, transfer: bool, exp: str) -> Tuple[List[float], List[float], float]:
        results_acc = defaultdict(list)
        results_loss = defaultdict(list)

        orgModel = org_layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes, self.data.orgGraph.num_nodes, self.emb_dim, len(self.data.sumGraphs))
        if embedding_trick != None:
            embedding_trick(self.data.orgGraph, self.data.sumGraphs, self.emb_dim)
            orgModel.load_embedding(self.data.orgGraph.training_data.embedding.clone())

        if transfer == True:
            self.transfer_weights(orgModel)
        print('Training on Orginal Graph...')
        results_acc[f'{exp} Accuracy'], results_loss[f'{exp} Loss'] = self.train(orgModel, self.data.orgGraph, sum_graph=False)

        # evaluatte on Test set
        pred = orgModel(self.data.orgGraph.training_data)
        pred = torch.round(pred)
        test_acc = self.calc_acc(pred, self.data.orgGraph.training_data.x_test, self.data.orgGraph.training_data.y_test)
        print('ACCURACY ON TEST SET = ',  test_acc)
        return results_acc, results_loss, test_acc
