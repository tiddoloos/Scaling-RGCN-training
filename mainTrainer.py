from helpers.graphData import Dataset
from helpers.RGCN import RGCN
from helpers.createMappings import invert_dict
from helpers.clean_createMapping import main
import torch

class modelTrainer:
    data = Dataset('AIFB')
    data.init_dataset()
    def __init__(self, hidden_l):
        self.hidden_l = hidden_l
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sumModel = RGCN(self.data.sumGraph.num_nodes, len(self.data.sumGraph.relations.keys()), self.hidden_l, self.data.num_labels)
        self.orgModel = RGCN(self.data.orgGraph.num_nodes, len(self.data.orgGraph.relations.keys()), self.hidden_l, self.data.num_labels)
        self.sumData = self.data.sum_training_data.to(self.device)
        self.orgData = self.data.org_training_data.to(self.device)
        self.accuracies = []

    def calc_acc(self, pred, x, y):
        tot = torch.sum(y == 1).item()
        p = (torch.sum((pred[x] == y) * (pred[x] == 1))) / tot
        return p.item()
    
    def evaluate(self, model, edge_index, edge_type):
        pred = model(edge_index, edge_type)
        pred = torch.round(pred)
        acc = self.calc_acc(pred, self.orgData.x_val, self.orgData.y_val)
        self.accuracies.append(acc)
        print(f'Accuracy on validation set = {acc}')

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
            targets = training_data.y.to(torch.float32)
            output = loss_f(out[training_data.idx], targets)
            output.backward()
            optimizer.step()
            l = output.item()
            if not sum_graph:
                model.eval()
                self.evaluate(model, graph.edge_index, graph.edge_type)
            print(f'Epoch: {epoch}, Loss: {l:.4f}')
        return

    def transfer_weights(self):
        print('...transferring weights...')

        # refactor this in mapping creation
        o_to_sg = invert_dict(self.data.sum2orgNode)
    
        weight_sg_1 = torch.rand(len(self.data.sumGraph.relations.keys()), self.data.orgGraph.num_nodes, self.hidden_l)
        root_sg_1 = torch.rand((self.data.orgGraph.num_nodes, self.hidden_l))

        # rgcn1
        for node in self.data.orgGraph.nodes:
            node = str(node).lower()
            if o_to_sg.get(node) != None:
                o_node_idx = self.data.orgGraph.node_to_enum[node]
                sg_node_idx = self.data.sumGraph.node_to_enum[o_to_sg[node][0]]
                weight_sg_1[:, o_node_idx, :] = self.sumModel.rgcn1.weight[:, sg_node_idx, :]
                root_sg_1[o_node_idx, :] = self.sumModel.rgcn1.root[sg_node_idx, :]
        bias_sg_1 = self.sumModel.rgcn1.bias
        
        # rgcn2
        weight_sg_2 = self.sumModel.rgcn2.weight
        root_sg_2 = self.sumModel.rgcn2.root
        bias_sg_2 = self.sumModel.rgcn2.bias

        # # actual transferring
        self.orgModel.override_params(weight_sg_1, bias_sg_1, root_sg_1,
                                            weight_sg_2, bias_sg_2, root_sg_2)
        print('transfer done!')

    def main_training(self, benchmark =False):
        epochs = 50
        weight_d = 0.0005
        lr = 0.01
        #train on sumModel
        if benchmark:
            self.train(self.orgModel, lr, weight_d, epochs)
            return
        else:
            self.train(self.sumModel, lr, weight_d, epochs, sum_graph=True)
            #transfer weights
            self.transfer_weights()
            #train orgModel
            self.train(self.orgModel, lr, weight_d, epochs)

main('AIFB')
# trainer_trans = modelTrainer(hidden_l=16)
# trainer_trans.main_training(benchmark=False)
# trainer_bench = modelTrainer(hidden_l=16)
# trainer_bench.main_training(benchmark=True)
