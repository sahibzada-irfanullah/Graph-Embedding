from RandomWalkEmbedding import RandomWalkEmbedding
import torch
import random
class DeepWalk(RandomWalkEmbedding):
    # constructor
    def __init__(self, graph, walkLength, embedDim, numbOfWalksPerVertex, windowSize, lr):
        super(DeepWalk, self).__init__(graph, walkLength, embedDim, numbOfWalksPerVertex)
        self.walkLength = walkLength
        self.windowSize = windowSize
        self.lr = lr
        self.model = None

    # Walks generation for deepwalk method
    def RandomWalk(self, node,t):
        walk = [int(self.nodeEncoder.transform([node]))]        # Walk starts from node "node"
        for i in range(t-1):
            neighborsList = [n for n in self.graph.neighbors(node)]
            node = neighborsList[random.randint(0,len(neighborsList)-1)]
            walk.append(int(self.nodeEncoder.transform([node])))
        return walk

    # Generate features for nodes
    def generateNodeFeatures(self, totalNodes, wvi, j):
        nodeFeatures = torch.zeros(totalNodes)
        nodeFeatures[wvi[j]] = 1
        return nodeFeatures

    # Training graph embedding model
    def learnEmbedding(self, model, wvi):
        for j in range(len(wvi)):
            for k in range(max(0,j-self.windowSize) , min(j+self.windowSize, len(wvi))):
                # Getting node for a specific features
                nodeFeatures = self.generateNodeFeatures(self.totalNodes, wvi, j)
                out = self.model.forward(nodeFeatures)
                loss = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
                loss.backward()
                for param in self.model.parameters():
                    param.data.sub_(self.lr*param.grad)
                    param.grad.data.zero_()
        return self.model

    # Training node embedding model
    def learnNodeEmbedding(self, model):
        self.model = model
        nodesList = list(self.graph.nodes)
        # Number of walks for a single vertex
        for i in range(self.numbOfWalksPerVertex):
            random.shuffle(nodesList)
            # Generating walk for a vertex
            for vi in nodesList:
                wvi= self.RandomWalk(vi, self.walkLength)
                self.model = self.learnEmbedding(self.model, wvi)
        return self.model

    # Get node embedding for a specific node, i.e., "node"
    def getNodeEmbedding(self, node):
        return self.model.W1[node].data

    # Initiate Training
    def learnEdgeEmbedding(self, model):
        self.model = model
        for startNode in list(self.graph.nodes):
            # Number of walks for a single vertex
            for i in range(self.numbOfWalksPerVertex):
                # Generating walk for a vertex
                walkStartNode = self.RandomWalk(startNode, self.walkLength)
                self.model = self.learnEmbedding(walkStartNode)
        return self.model
    # Get edge embedding for a specific edge having source node, i.e., "srcNode" and destination node, i.e., dstNode
    def getEdgeEmbedding(self, srcNode, dstNode):
        return self.operator_hadamard(self.getNodeEmbedding(srcNode), self.getNodeEmbedding(dstNode))

    def operator_hadamard(self, u, v):
        return u * v