from skipgram import SkipGramModel
from DeepWalk import DeepWalk
import os
import networkx as nx
import pandas as pd


import sys


print("User Current Version:-", sys.version)

dataset = "cora"
data_dir = "../cora"
data_dir = os.path.expanduser(data_dir)
edgelist = pd.read_csv(os.path.join(data_dir, dataset + ".cites"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "cites"

# input graph
my_graph = nx.from_pandas_edgelist(edgelist, edge_attr="label")


embedDim = 2 # embedding size
numbOfWalksPerVertex = 2 # walks per vertex
walkLength = 4 # walk lenght
lr =0.025 # learning rate
windowSize = 3 # window size


# DeepWalk
dw = DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)
# Skip Gram model
model_skip_gram = SkipGramModel(dw.totalNodes, dw.embedDim)

# Learning Node Embedding
model = dw.learnNodeEmbedding(model_skip_gram)

# Plot Embedding
# plot_2DEmbedding(dw)

# Get Embedding for a simple node
print(dw.getNodeEmbedding(10))
