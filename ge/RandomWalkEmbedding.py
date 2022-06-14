from abc import ABC, abstractmethod
from sklearn import preprocessing
class RandomWalkEmbedding:
    def __init__(self, graph, walkLength, embedDim, numbOfWalksPerVertex):
        self.graph = graph
        self.walkLength = walkLength
        self.embedDim = embedDim
        self.numbOfWalksPerVertex = numbOfWalksPerVertex
        self.adj_list, self.nodeEncoder = self.graph_to_adjList(graph)
        self.totalNodes = graph.number_of_nodes()
    #         self.nodesList = list(self.nodeEncoder.transform(list(graph.nodes)))
    #         self.nodesList = list(graph.nodes)


    def encoder(self, graph):
        nodeEncoder = preprocessing.LabelEncoder()
        return nodeEncoder.fit(list(graph.nodes()))

    def graph_to_adjList(self, graph):
        nodeEncoder = self.encoder(graph)
        adj_list1 = [None] * graph.number_of_nodes()
        for node, edges in list(graph.adjacency()):

            #     print(node, list(edges.keys()))
            adj_list1[nodeEncoder.transform([node])[0]] = list(nodeEncoder.transform(list(edges.keys())))
        return adj_list1, nodeEncoder

    def getAdjacencyList(self):
        return self.adj_list

    def getGraph(self):
        return self.graph


    @abstractmethod
    def generateWalk(self):
        pass

    @abstractmethod
    def learnEmbedding(self):
        pass

    @abstractmethod
    def learnNodeEmbedding(self):
        pass


    @abstractmethod
    def getNodeEmbedding(self):
        pass

    @abstractmethod
    def learnEdgeEmbedding(self):
        pass


    @abstractmethod
    def getEdgeEmbedding(self):
        pass