import torch
from typing import Callable, Tuple
from torch.nn import BCELoss, CrossEntropyLoss, Softmax

def get_functions(dataset, sumModel=False) -> Tuple[Callable]:
    if sumModel or dataset == 'AIFB':
        return BCELoss(), torch.sigmoid
    else:
        return CrossEntropyLoss(), Softmax(dim=1)