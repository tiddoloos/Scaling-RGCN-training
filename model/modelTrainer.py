from helpers.graphData import Dataset, Graph
from model.layers import transfer_Layers, mlp_RGCN_Layers, attention_Layers, baseline_Layers
import torch
from torch import Tensor
from typing import List, Tuple, Dict


class modelTrainer:
    def __init__(self, name, hidden_l: int):
        self.data = Dataset(name)
        self.data.init_dataset()
        self.hidden_l = hidden_l
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sumModel = None
        self.orgModel = None
        self.embModel = None
        self.benchModel = baseline_Layers(self.data.orgGraph.num_nodes, len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes)

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
    
    def evaluate(self, model, edge_index, edge_type) -> float:
        pred = model(edge_index, edge_type)
        pred = torch.round(pred)
        acc = self.calc_acc(pred, self.data.orgGraph.training_data.x_val, self.data.orgGraph.training_data.y_val)
        print(f'Accuracy on validation set = {acc}')
        return acc

    def train(self, model: transfer_Layers, graph: Graph, lr: float, weight_d: float, epochs: int, sum_graph=True) -> Tuple[List, List]:
        #initialize embedding 

        training_data = graph.training_data.to(self.device)
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
            if epoch%10==0:
                print(f'Epoch: {epoch}, Loss: {l:.4f}')
        
        return accuracies, losses

    def main_modelTrainer(self, epochs: int, weight_d: float, lr: float, exp)-> Dict[str, List[float]]:
        results = dict()
        
        if exp == 'baseline':
            print('--BASELINE EXP TRAINING--')
            results['Baseline Accuracy'], results['Baseline Loss'] = self.train(self.benchModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            return results
        
        if exp == 'embedding':
            print('--EMBEDDING EXP TRAINING--')
            self.embModel = transfer_Layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes)
            self.embModel.init_embeddings(self.data.orgGraph.num_nodes)
            results['Embedding Accuracy'], results['Embedding Loss'] = self.train(self.embModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            return results
        
        if exp == 'transfer':
            
            #train sum model
            print('---TRANSFER EXP TRAINING--')
            count = 0  
            self.sumModel = transfer_Layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes)
            print('...Training on Summary Graphs...')
            for sum_graph in self.data.sumGraphs:
                self.sumModel.init_embeddings(sum_graph.num_nodes)
                _, results[f'Sum Loss {count}'] = self.train(self.sumModel, sum_graph, lr, weight_d, epochs)
                #save embeddings in grpah object
                sum_graph.embedding = self.sumModel.embedding
                count += 1
            
            self.orgModel = transfer_Layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes)
            #make embedding for orgModel by summing
            self.orgModel.sum_embeddings(self.data.orgGraph, self.data.sumGraphs)

            #transfer weights
            self.transfer_weights()

            #train orgModel
            print('...Training on Orginal Graph after transfer...')
            results['Transfer Accuracy'], results['Transfer Loss'] = self.train(self.orgModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            return results
        
        if exp == 'mlp':
            #train sum model
            print('---MLP EMBEDDING EXP TRAINING--')
            count = 0  
            self.sumModel = transfer_Layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes)
            print('...Training on Summary Graphs...')
            for sum_graph in self.data.sumGraphs:
                self.sumModel.init_embeddings(sum_graph.num_nodes)
                _, results[f'Sum Loss {count}'] = self.train(self.sumModel, sum_graph, lr, weight_d, epochs)
                #save embeddings in grpah object
                sum_graph.embedding = self.sumModel.embedding
                count += 1
            
            #make embedding for orgModel by summing
            self.orgModel = mlp_RGCN_Layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes)
            self.orgModel.concat_embeddings(self.data.orgGraph, self.data.sumGraphs)

            #transfer weights
            self.transfer_weights()

            #train orgModel
            print('...Training on Orginal Graph after transfer...')
            results['Transfer Accuracy'], results['Transfer Loss'] = self.train(self.orgModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            return results

