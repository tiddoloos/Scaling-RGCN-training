import torch
from torch import nn
from helpers.graphData import Dataset, Graph
from model.models import emb_layers, emb_mlp_Layers, emb_att_Layers, base_Layers
from torch import Tensor
from typing import List, Tuple, Dict
from model.embeddingUtils import stack_embeddings, sum_embeddings, concat_embeddings


class modelTrainer:
    def __init__(self, name, hidden_l: int):
        self.data = Dataset(name)
        self.data.init_dataset()
        self.hidden_l = hidden_l
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.baseModel = base_Layers(self.data.orgGraph.num_nodes, len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes)
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
        #initialize embedding 

        training_data = graph.training_data.to(self.device)
        loss_f = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_d)
        accuracies = []
        losses = []

        for epoch in range(epochs):
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
                model.eval()
                acc = self.evaluate(model, graph)
                accuracies.append(acc)
            if epoch%10==0:
                print(f'Epoch: {epoch}, Loss: {l:.4f}')
                if not sum_graph:
                    print(f'Accuracy on validation set = {acc}')
        
        return accuracies, losses
    
    def print_trainable_parameters(self, model, exp: str) -> int:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'number of trainable parameters for {exp.upper()} model: {trainable_params}')
        return trainable_params
    
    def init_sumgraph_embeddings(self, emb_dim):
        for sum_graph in self.data.sumGraphs:
            sum_graph.embedding = nn.Embedding(sum_graph.num_nodes, emb_dim)

    def main_modelTrainer(self, epochs: int, weight_d: float, lr: float, emb_dim: int, exp: str)-> Dict[str, List[float]]:
        results_acc = dict()
        results_loss = dict()
        
        if exp == 'baseline':
            print('--BASELINE EXP TRAINING--')
            results_acc['Baseline Accuracy'], results_loss['Baseline Loss'] = self.train(self.baseModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            self.print_trainable_parameters(self.baseModel, exp)
            return results_acc, results_loss

 #####################################################################################################################################

        if exp == 'embedding':
            print('--EMBEDDING EXP TRAINING--')
            self.embModel = emb_layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes, emb_dim)
            self.data.orgGraph.embedding = nn.Embedding(self.data.orgGraph.num_nodes, emb_dim)
            results_acc['Embedding Accuracy'], results_loss['Embedding Loss'] = self.train(self.embModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            self.print_trainable_parameters(self.embModel, exp)
            return results_acc, results_loss

#####################################################################################################################################

        if exp == 'sum':

            #train summary model
            print('---TRANSFER SUM EXP TRAINING--')
            count = 0  
            self.sumModel = emb_layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes, emb_dim)
            self.init_sumgraph_embeddings(emb_dim)

            print('...Training on Summary Graphs...')
            for sum_graph in self.data.sumGraphs:
                _, results_loss[f'Sum Loss {count}'] = self.train(self.sumModel, sum_graph, lr, weight_d, epochs)
                count += 1

            self.orgModel = emb_layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes, emb_dim)
            #make embedding for orgModel by summing
            sum_embeddings(self.data.orgGraph, self.data.sumGraphs, emb_dim)
            
            #transfer weights
            self.transfer_weights()

            #train orgModel
            print('...Training on Orginal Graph after transfer...')
            results_acc['Transfer + sum Accuracy'], results_loss['Transfer + sum Loss'] = self.train(self.orgModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            self.print_trainable_parameters(self.orgModel, exp)

            return results_acc, results_loss
        

#####################################################################################################################################


        if exp == 'mlp':
            print('---MLP EMBEDDING EXP TRAINING--')
            self.sumModel = emb_layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes, emb_dim)
            self.init_sumgraph_embeddings(emb_dim)
            count = 0  
            #train summary model
            print('...Training on Summary Graphs...')
            for sum_graph in self.data.sumGraphs:
                _, results_loss[f'Sum Loss {count}'] = self.train(self.sumModel, sum_graph, lr, weight_d, epochs)
                count += 1
            
            in_f = len(self.data.sumGraphs)*emb_dim
            out_f = round((in_f/2)*3 + self.data.num_classes)
            self.orgModel = emb_mlp_Layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes, in_f, out_f, emb_dim)
             #make embedding for orgModel by concatinating
             
            concat_embeddings(self.data.orgGraph, self.data.sumGraphs, emb_dim)

            #transfer weights
            self.transfer_weights()

            #train orgModel
            print('...Training on Orginal Graph after transfer...')
            results_acc['Transfer + mlp Accuracy'], results_loss['Transfer + mlp Loss'] = self.train(self.orgModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
            self.print_trainable_parameters(self.orgModel, exp)

            return results_acc, results_loss

#####################################################################################################################################


        if exp == 'attention':
            print('---ATTENTION EMBEDDING EXP TRAINING--')
            self.sumModel = emb_layers(len(self.data.sumGraphs[0].relations.keys()), self.hidden_l, self.data.num_classes, emb_dim)
            self.init_sumgraph_embeddings(emb_dim)

            print('...Training on Summary Graphs...')
            #train sum model
            count = 0  
            for sum_graph in self.data.sumGraphs:
                _, results_loss[f'Sum Loss {count}'] = self.train(self.sumModel, sum_graph, lr, weight_d, epochs)
                #save embeddings in grpah object
                count += 1
            
            self.orgModel = emb_att_Layers(len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_classes, len(self.data.sumGraphs), emb_dim)
            #stack embeddings to use in attention layer
            stack_embeddings(self.data.orgGraph, self.data.sumGraphs, emb_dim)

            #transfer weights
            self.transfer_weights()

            #train orgModel
            print('...Training on Orginal Graph after transfer...')
            results_acc['Transfer + Attention Accuracy'], results_loss['Transfer + Attention Loss'] = self.train(self.orgModel, self.data.orgGraph, lr, weight_d, epochs, sum_graph=False)

            # in this calculation the parameters for making/updating the emedding do count aswell
            self.print_trainable_parameters(self.data.sumGraphs[0].embedding, 'Embedding')
            self.print_trainable_parameters(self.orgModel, exp)

            return results_acc, results_loss

