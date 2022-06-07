from skipgram import SkipGramModel
from DeepWalk import DeepWalk
from plotting import plot_2DEmbedding
from Node2Vec import Node2vec
import os
import networkx as nx
import pandas as pd


# dataset = "cora"
# data_dir = "../cora"

dataset = "citeseer"
data_dir = "../citeseer/"
data_dir = os.path.expanduser(data_dir)
edgelist = pd.read_csv(os.path.join(data_dir, dataset + ".cites"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "cites"

# input graph
my_graph = nx.from_pandas_edgelist(edgelist, edge_attr="label")


embedDim = 20 # embedding size
numbOfWalksPerVertex = 12 # walks per vertex
walkLength = 14 # walk lenght
lr =0.025 # learning rate
windowSize = 5 # window size


# DeepWalk
# dw = DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
#               windowSize=windowSize, lr = lr)

dw = Node2vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr=lr, p = 0.5, q = 0.8)

# Skip Gram model
model_skip_gram = SkipGramModel(dw.totalNodes, dw.embedDim)

# Learning Node Embedding
model = dw.learnNodeEmbedding(model_skip_gram)

# Plot Embedding
plot_2DEmbedding(dw)

# Get Embedding for a simple node
print(dw.getNodeEmbedding(10))
