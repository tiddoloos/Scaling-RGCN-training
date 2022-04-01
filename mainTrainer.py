from helpers.graphData import Dataset
from helpers.RGCN import RGCN
import torch

class modelTrainer:
    data = Dataset('AIFB')
    data.init_dataset()
    def __init__(self, hidden_l):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sumModel = RGCN(self.data.sumGraph.num_nodes, len(self.data.sumGraph.relations.keys()), hidden_l, self.data.num_labels)
        self.orgModel = RGCN(self.data.orgGraph.num_nodes, len(self.data.orgGraph.relations.keys()), hidden_l, self.data.num_labels)
        self.sumData = self.data.sum_training_data.to(self.device)
        self.orgData = self.data.org_training_data.to(self.device)

    def train(self, model, lr, weight_d, epochs, sum_graph=False):
        training_data = self.orgData
        graph = self.data.orgGraph
        if sum_graph:
            training_data = self.sumData
            graph = self.data.sumGraph
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_d)
        loss_f = torch.nn.BCELoss()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(graph.edge_index, graph.edge_type)
            targets = training_data.y_train.to(torch.float32)
            output = loss_f(out[training_data.idx], targets)
            output.backward()
            optimizer.step()
            l = output.item()
            # if not sum_graph:
            #     model.eval()
            print(f'Epoch: {epoch}, Loss: {l:.4f}')
        return

    def main_training(self):
        epochs = 50
        weight_d = 0.0005
        lr = 0.01
        self.train(self.sumModel, lr, weight_d, epochs, sum_graph=True)
        #transfer weights
        #train orgmodel

trainer = modelTrainer(hidden_l=16)
trainer.main_training()
