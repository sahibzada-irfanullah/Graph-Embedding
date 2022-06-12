from skipgram import SkipGramModel
from DeepWalk import DeepWalk
from plotting import plot_2DEmbedding
from Node2Vec import Node2vec
import os
import networkx as nx
import pandas as pd


# dataset = "cora"
# data_dir = "../cora"
#
# # dataset = "citeseer"
# # data_dir = "../citeseer/"
# data_dir = os.path.expanduser(data_dir)
# edgelist = pd.read_csv(os.path.join(data_dir, dataset + ".cites"), sep='\t', header=None, names=["target", "source"])
# edgelist["label"] = "cites"
#
# # input graph
# my_graph = nx.from_pandas_edgelist(edgelist, edge_attr="label")
my_graph = nx.Graph()
my_graph.add_edges_from([(10,11),(10,12),(10,13),
                         (11,10),(11,12),(11,13),
                         (12,10),(12,11),(12,13),
                         (13,10),(13,11),(13,12),
                         (14,15),(14,16),
                         (15,14),(15,16),
                         (16,14),(16,16),
                         (17,11),(17,13)])


embedDim = 2 # embedding size
numbOfWalksPerVertex = 2 # walks per vertex
walkLength = 4 # walk lenght
lr =0.25 # learning rate
windowSize = 3 # window size


# DeepWalk
# dw = DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
#               windowSize=windowSize, lr = lr)

dw = Node2vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr=lr, p = 0.5, q = 0.8)

# Skip Gram model
model_skip_gram = SkipGramModel(dw.totalNodes, dw.embedDim)

# Learning Node Embedding
model = dw.learnNodeEmbedding(model_skip_gram)

# Learning Node Embedding
model = dw.learnEdgeEmbedding(model_skip_gram)

# Plot Embedding
plot_2DEmbedding(dw)

# Get Embedding for a simple node
print(dw.getNodeEmbedding(list(my_graph.nodes())[4]))
