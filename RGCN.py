import torch
from torch import nn, sigmoid
from torch_geometric.nn import RGCNConv
from torch import functional as F

class RGCN(nn.Module):
    def __init__(self, in_channels, out_channels, dim_output, num_relations):
        super(RGCN, self).__init__()
        self.rgcn1 = RGCNConv(in_channels=in_channels, out_channels=out_channels, num_relations=num_relations)
        self.rgcn2 = RGCNConv(in_channels=out_channels, out_channels=dim_output, num_relations=num_relations)

    def forward(self, inp):
        x = self.rgcn1(inp)
        x = self.rgcn2(x)
        output = F.sigmoid(x)
        return output