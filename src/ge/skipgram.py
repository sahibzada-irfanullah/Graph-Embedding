import torch
import torch.nn as nn
class SkipGramModel(torch.nn.Module):
    # Constructor
    def __init__(self, totalNodes, embedDim):
        super(SkipGramModel, self).__init__()
        self.layers(totalNodes, embedDim)

    # Layers for a skipgram model
    def layers(self, totalNodes, embedDim):
        self.W1  = nn.Parameter(torch.rand((totalNodes, embedDim), requires_grad=True))
        self.W2 = nn.Parameter(torch.rand((embedDim, totalNodes), requires_grad=True))

    # Training skipgram model
    def forward(self, features):
        hidden = torch.matmul(features, self.W1)
        embed = torch.matmul(hidden, self.W2)
        # Get embedding
        return embed