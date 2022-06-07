# python -m cProfile -s time my_app.py <args>

import cProfile
import pstats
import io

from main import initialize_expirements
from model.models import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings

experiments = {
'sum': {'sum_layers': Emb_Layers, 'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
'mlp': {'sum_layers': Emb_Layers, 'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
'attention': {'sum_layers': Emb_Layers, 'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True},
'baseline': {'sum_layers': None, 'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False}
}

args = {'dataset': 'AIFB', 'exp': None, 'i': 1, 'hl': 16, 'epochs': 51, 'emb': 63, 'lr': 0.01}

pr = cProfile.Profile()
pr.enable()

my_result = initialize_expirements(args, experiments)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('performance/cProfile_out.txt', 'w+') as f:
    f.write(s.getvalue())