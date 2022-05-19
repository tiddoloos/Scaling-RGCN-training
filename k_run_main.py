
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings
from model.models import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from main import initialize_expirements

"""use this file to run experiments in main.py multiple times en get an average result for accuracy and loss"""

def run_k_times(k, args, experiments):
    for i in k:
        initialize_expirements(args, experiments)
    return



experiments = {
'sum': {'sum_layers': Emb_Layers, 'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
'mlp': {'sum_layers': Emb_Layers, 'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
'attention': {'sum_layers': Emb_Layers, 'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True},
'baseline': {'sum_layers': None, 'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False}
}

k=3
args = {'dataset': 'AIFB', 'exp': None}

