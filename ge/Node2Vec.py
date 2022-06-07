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

    def RandomWalk(self, startNode, walkLength):
        # walk contains encoded node labels
        walk = [int(self.nodeEncoder.transform([startNode]))]
        walkOptions = list(self.graph[startNode])
        if len(walkOptions)==0:
            return walk
        firstStep = np.random.choice(walkOptions)
        walk.append(int(self.nodeEncoder.transform([firstStep])))
        for k in range(walkLength-2):
            walkOptions = list(self.graph[(self.nodeEncoder.inverse_transform([walk[-1]])[0])])
            if len(walkOptions)==0:
                break
            probs = self.computeProbabilities((self.nodeEncoder.inverse_transform([walk[-2]])[0]))
            probabilities = probs[(self.nodeEncoder.inverse_transform([walk[-2]])[0])] \
                ['probabilities'][(self.nodeEncoder.inverse_transform([walk[-1]])[0])]
            nextStep = np.random.choice(walkOptions, p=probabilities)
            walk.append(int(self.nodeEncoder.transform([nextStep])))
        return walk

