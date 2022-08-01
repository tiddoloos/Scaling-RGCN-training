
import torch
from torch import Tensor, nn
from typing import Tuple, Callable
from torch_geometric.data import Data
from sklearn.metrics import classification_report, f1_score, accuracy_score

def calc_acc(pred: Tensor, x: Tensor, y: Tensor) -> float:
    return accuracy_score(y, pred[x])

def calc_f1(pred: Tensor, x: Tensor, y: Tensor, avg: str='weighted') -> float:
    return f1_score(y, pred[x], average=avg, zero_division=0)

def evaluate(model: nn.Module, activation, traininig_data: Data, x: Tensor, y: Tensor, report=False) -> Tuple[float]:
    pred = model(traininig_data, activation)
    if activation != torch.sigmoid:
        softmax = nn.Softmax(dim=1)
        pred = softmax(pred)
        a = pred.argmax(1)
        pred = torch.zeros(pred.shape).scatter (1, a.unsqueeze (1), 1.0)
    else:
        pred = torch.round(pred)
        pred = pred.type(torch.int64)
    acc = calc_acc(pred, x, y)
    f1_w = calc_f1(pred, x, y)
    f1_m = calc_f1(pred, x, y, avg='macro')

    if report:
        skl_pred = pred[x].detach().numpy()
        print(classification_report(y, skl_pred, zero_division=0))
    return acc, f1_w, f1_m

def ce_loss(pred: Tensor, targets: Tensor) -> Tensor:
    loss_f = nn.CrossEntropyLoss()
    targets = targets.argmax(-1)
    output = loss_f(pred, targets)
    return output

def bce_loss(pred: Tensor, targets: Tensor) -> Tensor:
    loss_f = nn.BCELoss()
    output = loss_f(pred, targets)
    return output

def get_losst(dataset: str, sumModel: bool=False) -> Tuple[Callable]:
    if sumModel or dataset == 'AIFB':
        return bce_loss, torch.sigmoid
    else:
        return ce_loss, do_nothing

def do_nothing(x: Tensor) -> Tensor:
    return x
