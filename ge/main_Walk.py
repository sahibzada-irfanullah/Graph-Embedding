from skipgram import SkipGramModel
from DeepWalk import DeepWalk
import os
import networkx as nx
import pandas as pd
from plotting import plot_2DEmbedding

dataset = "cora"
data_dir = "../cora"
# dataset = "citeseer"
# data_dir = "../citeseer/"
data_dir = os.path.expanduser(data_dir)
edgelist = pd.read_csv(os.path.join(data_dir, dataset + ".cites"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "cites"
my_graph = nx.from_pandas_edgelist(edgelist, edge_attr="label")


embedDim=2         # embedding size
numbOfWalksPerVertex = 2         # walks per vertex
walkLength = 4
lr =0.025
windowSize = 3

dw = DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)

model_skip_gram = SkipGramModel(dw.totalNodes, dw.embedDim)
# model = dw.learnEdgeEmbedding(model_skip_gram)
model = dw.learnNodeEmbedding(model_skip_gram)
plot_2DEmbedding(dw)
print(dw.getEdgeEmbedding(10, 11))
print(dw.getNodeEmbedding(10))
