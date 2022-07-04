from typing import Callable, Tuple
from torch.nn import BCELoss, CrossEntropyLoss

def get_function(dataset, sumModel=False) -> Tuple[Callable]:
    if sumModel or dataset == 'AIFB':
        return BCELoss()
    else:
        return CrossEntropyLoss()