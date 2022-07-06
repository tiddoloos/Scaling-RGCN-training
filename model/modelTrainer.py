import torch

from collections import defaultdict
from torch import nn
from typing import List, Tuple, Callable, Union, Dict, Callable

from graphdata.graph import Graph
from graphdata.dataset import Dataset
from model.layers import Emb_Layers
from model.evaluation import evaluate, get_losst

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

    def transfer_weights(self, orgModel: nn.Module, grad: bool) -> None:
        # rgcn1 
        weight_sg_1 = self.sumModel.rgcn1.weight.clone()
        bias_sg_1 = self.sumModel.rgcn1.bias.clone()
        root_sg_1 = self.sumModel.rgcn1.root.clone()

        # rgcn2
        weight_sg_2 = self.sumModel.rgcn2.weight.clone()
        bias_sg_2 = self.sumModel.rgcn2.bias.clone()
        root_sg_2 = self.sumModel.rgcn2.root.clone()

        # transfer
        orgModel.override_params(weight_sg_1, bias_sg_1, root_sg_1, weight_sg_2, bias_sg_2, root_sg_2, grad)
        print('weight transfer done')
    
    def train(self, model: nn.Module, graph: Graph, loss_f: Callable, activation: Callable, sum_graph: bool=True) -> Tuple[List[float]]:
        model = model.to(self.device)
        training_data = graph.training_data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_d)

        accuracies: list = []
        losses: list = []
        f1_ws: list = []
        f1_ms: list = []
        
        for epoch in range(self.epochs):

            if not sum_graph:
                model.eval()
                acc, f1_w, f1_m = evaluate(model, activation, training_data, training_data.x_val, training_data.y_val)
                print(f'Accuracy on validation set = {acc}')  
                accuracies.append(acc)
                f1_ws.append(f1_w)
                f1_ms.append(f1_m)
                # out = model(training_data, activation)
                # targets = training_data.y_val.to(torch.float32)
                # output = loss_f(out[training_data.x_val], targets)
                # loss = output.item()
                # losses.append(loss)

            model.train()
            optimizer.zero_grad()
            out = model(training_data, activation)
            targets = training_data.y_train.to(torch.float32)
            output = loss_f(out[training_data.x_train], targets)
            output.backward()
            optimizer.step()
            l = output.item()
            losses.append(l)
            if epoch%10==0:
                l = output.item()
                print(f'Epoch: {epoch}, Loss: {l:.4f}')
    
        return accuracies, losses, f1_ws, f1_ms

    def train_summaries(self, configs):
        loss_f, activation = get_losst(configs['dataset'], sumModel=True)
        self.sumModel = Emb_Layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes, self.data.sumGraphs[0].num_nodes, self.emb_dim, len(self.data.sumGraphs))
        for sumGraph in self.data.sumGraphs:
            self.sumModel.reset_embedding(sumGraph.num_nodes, self.emb_dim)
            _, _, _, _ = self.train(self.sumModel, sumGraph, loss_f, activation, sum_graph=True)
            sumGraph.embedding = self.sumModel.embedding.weight.clone()

    def train_original(self, org_layers: nn.Module, embedding_trick: Callable,
                        configs: Dict[str, Union[bool, str, int, float]], exp: str) -> Tuple[List[float], float]:

        acc = defaultdict(list)
        loss = defaultdict(list)
        f1_w = defaultdict(list)
        f1_m = defaultdict(list)

        orgModel = org_layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes, self.data.orgGraph.num_nodes, self.emb_dim, configs['num_sums'])
        
        if exp != 'baseline' and configs['e_trans'] == True:
            embedding = embedding_trick(self.data.orgGraph, self.data.sumGraphs, self.emb_dim)
            orgModel.load_embedding(embedding, freeze=configs["e_freeze"])
            print('Loaded pre trained embedding')

        if exp != 'baseline' and configs['w_trans'] == True:
            self.transfer_weights(orgModel, configs['w_grad'])

        loss_f, activation = get_losst(configs['dataset'], sumModel=False)
        
        print('Training on Orginal Graph...')
        acc[f'accuracy'], loss[f'loss'], f1_w[f'f1 weighted'], f1_m[f'f1 macro'] = self.train(orgModel, self.data.orgGraph, loss_f, activation, sum_graph=False)

        # evaluate on Test set
        test_acc, test_f1_weighted, test_f1_macro = evaluate(orgModel, activation, self.data.orgGraph.training_data, self.data.orgGraph.training_data.x_test, self.data.orgGraph.training_data.y_test, report=True)
        print('ACC ON TEST SET = ',  test_acc)
    
        return acc, loss, f1_w, f1_m, test_acc, test_f1_weighted, test_f1_macro
