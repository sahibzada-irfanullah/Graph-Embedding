# from struc2vec import Struc2Vec
import networkx as nx
from skipgram import SkipGramModel
from DeepWalk import DeepWalk
# from Node2Vec import Node2vec
import random
import os
import networkx as nx
import pandas as pd
# my_graph = nx.Graph()

# Add edges to to the graph object
# Each tuple represents an edge between two nodes
# my_graph.add_edges_from([(10,11),(10,12),(10,13),
#                          (11,10),(11,12),(11,13),
#                          (12,10),(12,11),(12,13),
#                          (13,10),(13,11),(13,12),
#                          (14,15),(14,16),
#                          (15,14),(15,16),
#                          (16,14),(16,16),
#                          (17,11),(17,13)])


# dataset = "cora"
# data_dir = "../cora"
dataset = "citeseer"
data_dir = "../citeseer/"
data_dir = os.path.expanduser(data_dir)
# print(os.path.join(data_dir, dataset + ".cites"))
edgelist = pd.read_csv(os.path.join(data_dir, dataset + ".cites"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "cites"
my_graph = nx.from_pandas_edgelist(edgelist, edge_attr="label")

# print("my_graph", my_graph.nodes)
embedDim=2         # embedding size
numbOfWalksPerVertex = 2         # walks per vertex
walkLength = 4
lr =0.025
windowSize = 3

# dw = Struc2Vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, windowSize = 3, lr = lr)
# dw = DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
#               windowSize=windowSize, lr = lr)

dw = Node2vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr=lr, p = 0.5, q = 0.8)

model_skip_gram = SkipGramModel(dw.totalNodes, dw.embedDim)
# model = dw.learnEdgeEmbedding(model_skip_gram)
model = dw.learnNodeEmbedding(model_skip_gram)
print(dw.getEdgeEmbedding(10, 11))
print(dw.getNodeEmbedding(10))
def plot_2DEmbedding(dw):
    import matplotlib.pyplot as plt
    xs = dw.model.W1.data[:, 0]
    ys = dw.model.W1.data[:, 1]
    ls = list(range(0, len(xs)))
    plt.scatter(xs, ys)
    for x,y,l in zip(xs,ys, ls):
        plt.annotate(str(int(dw.nodeEncoder.inverse_transform([l]))), (x, y))
    plt.show()
plot_2DEmbedding(dw)