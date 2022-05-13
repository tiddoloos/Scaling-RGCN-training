from torch import nn
from typing import List, Dict

from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings, init_sumgraph_embeddings
from model.modelTrainer import Trainer
from model.models import emb_layers, emb_mlp_Layers, emb_att_Layers


def print_trainable_parameters(model: nn.Module, exp: str) -> int:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of trainable parameters for {exp.upper()} model: {trainable_params}')
    return trainable_params

def run_experiment(trainer: Trainer, epochs: int, weight_d: float, lr: float, emb_dim: int, exp: str)-> Dict[str, List[float]]:
    results_acc = dict()
    results_loss = dict()
    
    if exp == 'baseline':
        print('--BASELINE EXP TRAINING--')
        results_acc['Baseline Accuracy'], results_loss['Baseline Loss'] = trainer.train(trainer.baseModel, trainer.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
        print_trainable_parameters(trainer.baseModel, exp)


    if exp == 'embedding':
        print('--EMBEDDING EXP TRAINING--')
        trainer.embModel = emb_layers(len(trainer.data.orgGraph.relations.keys()), trainer.hidden_l, trainer.data.num_classes, emb_dim)
        trainer.data.orgGraph.embedding = nn.Embedding(trainer.data.orgGraph.num_nodes, emb_dim)
        results_acc['Embedding Accuracy'], results_loss['Embedding Loss'] = trainer.train(trainer.embModel, trainer.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
        print_trainable_parameters(trainer.embModel, exp)


    if exp == 'sum':
        print('---TRANSFER SUM EXP TRAINING--')

        count = 0  
        trainer.sumModel = emb_layers(len(trainer.data.sumGraphs[0].relations.keys()), trainer.hidden_l, trainer.data.num_classes, emb_dim)
        init_sumgraph_embeddings(trainer, emb_dim)
        #train summary model
        print('...Training on Summary Graphs...')
        for sum_graph in trainer.data.sumGraphs:
            _, results_loss[f'Sum Loss {count}'] = trainer.train(trainer.sumModel, sum_graph, lr, weight_d, epochs)
            count += 1

        trainer.orgModel = emb_layers(len(trainer.data.orgGraph.relations.keys()), trainer.hidden_l, trainer.data.num_classes, emb_dim)
        #make embedding for orgModel by summing
        sum_embeddings(trainer.data.orgGraph, trainer.data.sumGraphs, emb_dim)
        
        #transfer weights
        trainer.transfer_weights()

        #train orgModel
        print('...Training on Orginal Graph after transfer...')
        results_acc['Transfer + sum Accuracy'], results_loss['Transfer + sum Loss'] = trainer.train(trainer.orgModel, trainer.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
        print_trainable_parameters(trainer.orgModel, exp)


    if exp == 'mlp':
        print('---MLP EMBEDDING EXP TRAINING--')
        trainer.sumModel = emb_layers(len(trainer.data.sumGraphs[0].relations.keys()), trainer.hidden_l, trainer.data.num_classes, emb_dim)
        init_sumgraph_embeddings(trainer, emb_dim)
        count = 0  
        #train summary model
        print('...Training on Summary Graphs...')
        for sum_graph in trainer.data.sumGraphs:
            _, results_loss[f'Sum Loss {count}'] = trainer.train(trainer.sumModel, sum_graph, lr, weight_d, epochs)
            count += 1
        
        in_f = len(trainer.data.sumGraphs)*emb_dim
        out_f = round((in_f/2)*3 + trainer.data.num_classes)
        trainer.orgModel = emb_mlp_Layers(len(trainer.data.orgGraph.relations.keys()), trainer.hidden_l, trainer.data.num_classes, in_f, out_f, emb_dim)
        
        #make embedding for orgModel by concatinating    
        concat_embeddings(trainer.data.orgGraph, trainer.data.sumGraphs, emb_dim)

        #transfer weights
        trainer.transfer_weights()

        #train orgModel
        print('...Training on Orginal Graph after transfer...')
        results_acc['Transfer + mlp Accuracy'], results_loss['Transfer + mlp Loss'] = trainer.train(trainer.orgModel, trainer.data.orgGraph, lr, weight_d, epochs, sum_graph=False)
        print_trainable_parameters(trainer.orgModel, exp)



    if exp == 'attention':
        print('---ATTENTION EMBEDDING EXP TRAINING--')
        trainer.sumModel = emb_layers(len(trainer.data.sumGraphs[0].relations.keys()), trainer.hidden_l, trainer.data.num_classes, emb_dim)
        init_sumgraph_embeddings(trainer, emb_dim)

        print('...Training on Summary Graphs...')
        #train sum model
        count = 0
        for sum_graph in trainer.data.sumGraphs:
            _, results_loss[f'Sum Loss {count}'] = trainer.train(trainer.sumModel, sum_graph, lr, weight_d, epochs)
            count += 1
        
        trainer.orgModel = emb_att_Layers(len(trainer.data.orgGraph.relations.keys()), trainer.hidden_l, trainer.data.num_classes, len(trainer.data.sumGraphs), emb_dim)

        #stack embeddings to use in attention layer
        stack_embeddings(trainer.data.orgGraph, trainer.data.sumGraphs, emb_dim)

        #transfer weights
        trainer.transfer_weights()

        #train orgModel
        print('...Training on Orginal Graph after transfer...')
        results_acc['Transfer + Attention Accuracy'], results_loss['Transfer + Attention Loss'] = trainer.train(trainer.orgModel, trainer.data.orgGraph, lr, weight_d, epochs, sum_graph=False)

        # in this calculation the parameters for making/updating the emedding do count aswell
        print_trainable_parameters(trainer.data.sumGraphs[0].embedding, 'Embedding')
        print_trainable_parameters(trainer.orgModel, exp)

    return results_acc, results_loss

