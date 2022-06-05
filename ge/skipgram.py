import torch
import torch.nn as nn
class SkipGramModel(torch.nn.Module):
    def __init__(self, totalNodes, embedDim):
        super(SkipGramModel, self).__init__()
        self.layers(totalNodes, embedDim)

    def layers(self, totalNodes, embedDim):
        self.W1  = nn.Parameter(torch.rand((totalNodes, embedDim), requires_grad=True))
        self.W2 = nn.Parameter(torch.rand((embedDim, totalNodes), requires_grad=True))

    def forward(self, one_hot):
        hidden = torch.matmul(one_hot, self.W1)
        out    = torch.matmul(hidden, self.W2)
        return out