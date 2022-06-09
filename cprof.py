# python -m cProfile -s time my_app.py <args>

import cProfile
import pstats
import io

from main import initialize_expirements
from model.layers import Emb_Layers, Emb_MLP_Layers, Emb_ATT_Layers
from model.embeddingTricks import stack_embeddings, sum_embeddings, concat_embeddings

"""this file tests the performance of all functions in the pipeline
results are saved as ./performance/cProfile_out.txt.
The file ranks functions on 'tottime' -> how much time the fuction is used in total
"""

methods = {'baseline': {
                'org_layers': Emb_Layers, 'embedding_trick': None, 'transfer': False},
            'experiments': {
                'summation': {'org_layers': Emb_Layers, 'embedding_trick': sum_embeddings, 'transfer': True},
                'mlp': {'org_layers': Emb_MLP_Layers, 'embedding_trick': concat_embeddings, 'transfer': True},
                'attention': {'org_layers': Emb_ATT_Layers, 'embedding_trick': stack_embeddings, 'transfer': True}}}



args = {'dataset': 'AIFB', 'exp': None, 'i': 1, 'hl': 16, 'epochs': 51, 'emb': 63, 'lr': 0.01}

pr = cProfile.Profile()
pr.enable()

my_result = initialize_expirements(args, methods, graph_pros_test=True)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('performance/cProfile_out.txt', 'w+') as f:
    f.write(s.getvalue())