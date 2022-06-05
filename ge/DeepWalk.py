from RandomWalkEmbedding import RandomWalkEmbedding
import torch
import torch.nn as nn
import random
class DeepWalk(RandomWalkEmbedding):
    def __init__(self, graph, walkLength, embedDim, numbOfWalksPerVertex, windowSize, lr):
        super(DeepWalk, self).__init__(graph, walkLength, embedDim, numbOfWalksPerVertex)
        self.walkLength = walkLength
        self.windowSize = windowSize
        self.lr = lr
        self.model = None

    def RandomWalk(self, node,t):
        walk = [int(self.nodeEncoder.transform([node]))]        # Walk starts from this node

        for i in range(t-1):
            neighborsList = [n for n in self.graph.neighbors(node)]
            node = neighborsList[random.randint(0,len(neighborsList)-1)]
            walk.append(int(self.nodeEncoder.transform([node])))
        return walk

    def generateNodeFeatures(self, totalNodes, wvi, j):
        #generate one hot vector
        nodeFeatures          = torch.zeros(totalNodes)
        #                 print(one_hot)
        nodeFeatures[wvi[j]]  = 1
        return nodeFeatures


    def learnEmbedding(self, model, wvi):
        for j in range(len(wvi)):
            for k in range(max(0,j-self.windowSize) , min(j+self.windowSize, len(wvi))):

                # #generate one hot vector
                # nodeFeatures          = torch.zeros(self.totalNodes)
                # #                 print(one_hot)
                # nodeFeatures[wvi[j]]  = 1
                nodeFeatures = self.generateNodeFeatures(self.totalNodes, wvi, j)

                out              = self.model.forward(nodeFeatures)
                loss             = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
                loss.backward()

                for param in self.model.parameters():
                    param.data.sub_(self.lr*param.grad)
                    param.grad.data.zero_()
        return self.model

    def learnNodeEmbedding(self, model):
        self.model = model
        nodesList = list(self.graph.nodes)
        for i in range(self.numbOfWalksPerVertex):
            random.shuffle(nodesList)
            for vi in nodesList:
                wvi= self.RandomWalk(vi, self.walkLength)
                self.model = self.learnEmbedding(self.model, wvi)
        return self.model

    def getNodeEmbedding(self, node):
        return self.model.W1[node].data
class SkipGramModel(torch.nn.Module):
    def __init__(self, totalNodes, embedDim):
        super(SkipGramModel, self).__init__()
        self.layers(totalNodes, embedDim)

    def layers(self, totalNodes, embedDim):
        if torch.cuda.is_available():
            self.W1  = nn.Parameter(torch.rand((totalNodes, embedDim))).to("cuda")
            self.W2 = nn.Parameter(torch.rand((embedDim, totalNodes))).to("cuda")
            print("GPU GPU", self.W1.requires_grad, self.W2.requires_grad, self.W1.is_cuda, self.W2.is_cuda)


    def forward(self, one_hot):
        if torch.cuda.is_available():
            one_hot = one_hot.to("cuda")
            # hidden = hidden.to("cuda")

        hidden = torch.matmul(one_hot, self.W1).to("cuda")
        out    = torch.matmul(hidden, self.W2)
        return out