import torch

from numpy import int64
from collections import defaultdict
from torch import nn, softmax
from torch import Tensor
from torch_geometric.data import Data
from typing import List, Tuple, Callable, Union, Dict, Callable
from sklearn.metrics import classification_report, f1_score, accuracy_score

from graphdata.graph import Graph
from graphdata.dataset import Dataset
from model.layers import Emb_Layers

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

    def calc_acc(self, pred: Tensor, x: Tensor, y: Tensor) -> float:
        # print(y[0])
        # print(pred[x][0])
        return accuracy_score(y, pred[x])

    def calc_f1(self, pred: Tensor, x: Tensor, y: Tensor, avg='weighted') -> float:
        return f1_score(y, pred[x], average=avg, zero_division=0)
    
    def evaluate(self, pred: Tensor, x: Tensor, y: Tensor, configs: dict, multi_label: bool, class_repo=False) -> float:
        # pred = model(traininig_data, activation)
        # targets = self.data.orgGraph.training_data.y_val.to(torch.float)
        output = self.calc_loss(x, y, pred, configs['dataset'], sumModel=False)
        loss = output.item()
        # print(multi_label)
        if multi_label:
            pred = torch.round(pred).detach().numpy()
        else:
            # print(pred)
            softmax = nn.Softmax(dim=1)
            pred = softmax(pred)
            # print(sum(pred[0]))
            max_idx = torch.argmax(pred, 0, keepdim=True)
            pred = torch.IntTensor(pred.shape)
            pred.zero_()
            pred.scatter_(0, max_idx, 1)
            # print(pred)

        acc = self.calc_acc(pred, x, y)
        f1_w = self.calc_f1(pred, x, y)
        f1_m = self.calc_f1(pred, x, y, avg='macro')
        if class_repo:
            classification_report(y, pred[x], zero_division=0)
        return acc, f1_w, f1_m, loss
    
    def calc_loss(self, x, y, out, dataset, sumModel=False):
        if sumModel or dataset == 'AIFB':
            loss_f = nn.BCELoss()
            targets = y.to(torch.float)
        else:
            loss_f = nn.CrossEntropyLoss() 
            targets = torch.argmax(y.to(torch.float), dim=1)
            # print(targets)
        output = loss_f(out[x], targets)
        # print(output)
        return output
    
    def train(self, model: nn.Module, graph: Graph, configs: dict, sumModel: bool=False) -> Tuple[List[float]]:

        multi_label = False
        if sumModel or configs['dataset'] == 'AIFB':
            multi_label = True

        model = model.to(self.device)
        training_data = graph.training_data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_d)

        accuracies: list = []
        losses: list = []
        f1_ws: list = []
        f1_ms: list = []

        for epoch in range(self.epochs):
            if not sumModel:
                model.eval()
                pred = model(training_data, multi_label)
                acc, f1_w, f1_m, loss = self.evaluate(pred, training_data.x_val, training_data.y_val, configs, multi_label)
                print(f'Accuracy on validation set = {acc}')  
                accuracies.append(acc)
                f1_ws.append(f1_w)
                f1_ms.append(f1_m)
                losses.append(loss)
            model.train()
            optimizer.zero_grad()
            pred = model(training_data, multi_label)
            
            output = self.calc_loss(training_data.x_train, training_data.y_train, pred, configs['dataset'], sumModel)

            # targets = training_data.y_train.to(torch.float)
            # output = loss_f(out[training_data.x_train], targets)
            output.backward()
            optimizer.step()
            l = output.item()
            if epoch%10==0:
                print(f'Epoch: {epoch}, Loss: {l:.4f}')
        return accuracies, losses, f1_ws, f1_ms

    def train_summaries(self, configs):
        self.sumModel = Emb_Layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes, self.data.sumGraphs[0].num_nodes, self.emb_dim, len(self.data.sumGraphs))
        for sumGraph in self.data.sumGraphs:
            self.sumModel.reset_embedding(sumGraph.num_nodes, self.emb_dim)
            _, _, _, _ = self.train(self.sumModel, sumGraph, configs, sumModel=True)
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
            orgModel.load_embedding(embedding)
            print('Loaded pre trained embedding')

        if exp != 'baseline' and configs['w_trans'] == True:
            self.transfer_weights(orgModel, configs['w_grad'])
        
        print('Training on Orginal Graph...')
        acc[f'accuracy'], loss[f'loss'], f1_w[f'f1 weighted'], f1_m[f'f1 macro'] = self.train(orgModel, self.data.orgGraph, configs, sumModel=False)

        # evaluate on Test set
        multi_label = False
        if configs['dataset'] == 'AIFB':
            multi_label = True

        pred = orgModel(self.data.orgGraph.training_data, multi_label)
        test_acc, test_f1_weighted, test_f1_macro, test_loss = self.evaluate(pred, self.data.orgGraph.training_data.x_test, self.data.orgGraph.training_data.y_test, configs, multi_label, class_repo=True)

        # pred = torch.round(pred)
        # pred = pred.type(torch.int64)
        # skl_pred = pred[self.data.orgGraph.training_data.x_test].detach().numpy()
        # print(classification_report(self.data.orgGraph.training_data.y_test, skl_pred, zero_division=0))

        # test_f1_weighted = self.calc_f1(pred, self.data.orgGraph.training_data.x_test, self.data.orgGraph.training_data.y_test)
        # test_f1_macro = self.calc_f1(pred, self.data.orgGraph.training_data.x_test, self.data.orgGraph.training_data.y_test, avg='macro')
        # test_acc = self.calc_acc(pred, self.data.orgGraph.training_data.x_test, self.data.orgGraph.training_data.y_test)
        print('ACC ON TEST SET = ',  test_acc)
    
        return acc, loss, f1_w, f1_m, test_acc, test_f1_weighted, test_f1_macro
