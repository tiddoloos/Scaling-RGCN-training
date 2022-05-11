import torch
from data.graphData import Dataset, Graph
from model.models import base_Layers
from torch import Tensor
from typing import List, Tuple


class modelTrainer:
    def __init__(self, name, hidden_l: int):
        self.device = torch.device(str('cuda:0') if torch.cuda.is_available() else 'cpu')
        self.data = Dataset(name)
        self.data.init_dataset()
        self.hidden_l = hidden_l
        self.baseModel = base_Layers(self.data.orgGraph.num_nodes, len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes).to(self.device)
        self.sumModel = None
        self.orgModel = None
        self.embModel = None

    def transfer_weights(self) -> None:
        # rgcn1 
        weight_sg_1 = self.sumModel.rgcn1.weight
        bias_sg_1 = self.sumModel.rgcn1.bias
        root_sg_1 = self.sumModel.rgcn1.root

        # rgcn2
        weight_sg_2 = self.sumModel.rgcn2.weight
        bias_sg_2 = self.sumModel.rgcn2.bias
        root_sg_2 = self.sumModel.rgcn2.root

        # transfer
        self.orgModel.override_params(weight_sg_1, bias_sg_1, root_sg_1, weight_sg_2, bias_sg_2, root_sg_2)
        print('weight transfer done')

    def calc_acc(self, pred: Tensor, x: Tensor, y: Tensor) -> float:
        tot = torch.sum(y == 1).item()
        p = (torch.sum((pred[x] == y) * (pred[x] == 1))) / tot
        return p.item()
    
    def evaluate(self, model, graph: Graph) -> float:
        pred = model(graph)
        pred = torch.round(pred)
        acc = self.calc_acc(pred, self.data.orgGraph.training_data.x_val, self.data.orgGraph.training_data.y_val)
        return acc

    def train(self, model, graph: Graph, lr: float, weight_d: float, epochs: int, sum_graph=True) -> Tuple[List, List]:
        model.to(self.device)
        loss_f = torch.nn.BCELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_d)
        training_data = graph.training_data.to(self.device)

        accuracies = []
        losses = []

        for epoch in range(epochs):
            if not sum_graph:
                model.eval()
                acc = self.evaluate(model, graph)
                accuracies.append(acc)
            model.train()
            optimizer.zero_grad()
            out = model(graph)
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