from RandomWalkEmbedding import RandomWalkEmbedding
import torch
import numpy as np
from collections import defaultdict

class Node2vec(RandomWalkEmbedding):
    def __init__(self, graph, walkLength, embedDim, numbOfWalksPerVertex, windowSize, lr, p, q):
        super(Node2vec, self).__init__(graph, walkLength, embedDim, numbOfWalksPerVertex)
        self.walkLength = walkLength
        self.windowSize = windowSize
        self.model = None
        self.lr = lr
        self.p = p
        self.q = q

