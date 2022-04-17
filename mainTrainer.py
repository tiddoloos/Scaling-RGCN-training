from helpers.graphData import Dataset, Graph
from helpers.RGCN import RGCN
import torch
from torch import Tensor
from helpers.plot import plot_main
from typing import List, Tuple

class modelTrainer:
    def __init__(self, name, hidden_l: int):
        self.data = Dataset(name)
        self.data.init_dataset()
        self.hidden_l = hidden_l
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sumModel = None
        self.orgModel = RGCN(self.data.orgGraph.num_nodes, len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes)
        self.benchModel = RGCN(self.data.orgGraph.num_nodes, len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes)
        # self.sumData = self.data.sum_training_data.to(self.device)
        # self.orgData = self.data.org_training_data.to(self.device)

    def transfer_weights(self) -> None:
        weight_sg_1 = torch.rand(len(self.data.sumGraphs[0].relations.keys()), self.data.orgGraph.num_nodes, self.hidden_l)
        root_sg_1 = torch.rand((self.data.orgGraph.num_nodes, self.hidden_l))

        # rgcn1
        for node in self.data.orgGraph.nodes:
            node = str(node).lower()
            if self.data.sumGraphs[0].orgNode2sumNode_dict.get(node) != None:
                o_node_idx = self.data.orgGraph.node_to_enum[node]
                sg_node_idx = self.data.sumGraphs[0].node_to_enum[self.data.sumGraphs[0].orgNode2sumNode_dict[node]]
                weight_sg_1[:, o_node_idx, :] = self.sumModel.rgcn1.weight[:, sg_node_idx, :]
                root_sg_1[o_node_idx, :] = self.sumModel.rgcn1.root[sg_node_idx, :]
        bias_sg_1 = self.sumModel.rgcn1.bias
        
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
    
    def evaluate(self, model, edge_index, edge_type) -> float:
        pred = model(edge_index, edge_type)
        pred = torch.round(pred)
        acc = self.calc_acc(pred, self.data.orgGraph.training_data.x_val, self.data.orgGraph.training_data.y_val)
        print(f'Accuracy on validation set = {acc}')
        return acc

    def train(self, model: RGCN, graph: Graph, lr: float, weight_d: float, epochs: int, sum_graph=True) -> Tuple[List, List]:
        training_data = graph.training_data        
        loss_f = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_d)
        accuracies = []
        losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(graph.edge_index, graph.edge_type)
            targets = training_data.y_train.to(torch.float32)
            output = loss_f(out[training_data.x_train], targets)
            output.backward()
            optimizer.step()
            l = output.item()
            losses.append(l)
            if not sum_graph:
                model.eval()
                accuracies.append(self.evaluate(model, graph.edge_index, graph.edge_type))
            print(f'Epoch: {epoch}, Loss: {l:.4f}')
        return accuracies, losses

    def main_modelTrainer(self, epochs: int, weight_d: float, lr: float, benchmark=False)-> Tuple[List[float], List[float], (List[float])]:
        results = dict()
        if benchmark:
            print('--BENCHMARK TRAINING ON ORIGINAL GRAPH--')
            results['Benchmark accuracy'], results['Benchmark loss'] = self.train(self.benchModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            return results
        else:
            #train sum model
            print('---TRAINING ON SUMMARY GRAPHS--')
            self.sumModel = RGCN(self.data.sumGraphs[0].num_nodes, len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes)
            count = 0
            for sum_graph in self.data.sumGraphs:
                
               _, results[f'sum loss {count}'] = self.train(self.sumModel, sum_graph, lr, weight_d, epochs)
               count += 1
        
            #transfer weights
            self.transfer_weights()
            #train orgModel
            print('--TRAINING ON ORIGINAL GRAPH--')
            results['Transfer accuracy'], results['Transfer Loss'] = self.train(self.orgModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            return results

def initialize_training() -> None:
    epochs = 51
    weight_d = 0.0005
    lr = 0.01
    hidden_l=16

    # Transfer learning expriment
    trainer = modelTrainer('AIFB', hidden_l)
    results_transfer = trainer.main_modelTrainer(epochs, weight_d, lr, benchmark=False)
    # results_dict['Summary graph loss'], results_dict['Orginal graph accuracy'], results_dict['Orginal graph loss'] = trainer_trans.main_modelTrainer(epochs, weight_d, lr, benchmark=False)

    # Benchmark
    results_benchmark = trainer.main_modelTrainer(epochs, weight_d, lr, benchmark=True)
    results = {**results_transfer, **results_benchmark}

    plot_main('AIFB', results, epochs)

initialize_training()
