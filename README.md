# Graph Embedding
### Introduction
This module provides the services and implementation for various
family of graph embedding algorithms.

### Installation

You can install the DGLL Graph Embedding version 1.0.0 from [PyPI](https://pypi.org/project/dgllge/) as:

    pip install dgllge

## How to utilize a code? (Tutorial)
#### input graph
```
# import module
import ge


# Set Path to Data
data_dir = "Your Path to Data"
dataset = "File/Dataset Name"


# Load a Graph
inputGraph = ge.loadGraph(data_dir, dataset)
```

#### Configurable Parameter for Graph Embedding
```
embedDim = 2 # embedding size
numbOfWalksPerVertex = 2 # walks per vertex
walkLength = 4 # walk lenght
lr =0.025 # learning rate
windowSize = 3 # window size
```

#### instantiating Graph Embedding Model
```
# choose of the following Graph embedding algorithm
# DeepWalk
rw = ge.DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)
              
              
# Node2Vec
rw = ge.Node2vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
               windowSize=windowSize, lr=lr, p = 0.5, q = 0.8)

# Struc2Vec
rw = ge.Struc2Vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)
              
# Skip Gram model
modelSkipGram = ge.SkipGramModel(rw.totalNodes, rw.embedDim)

# Choose whether want Node embedding or edge embedding
# Learning Node Embedding
model = rw.learnNodeEmbedding(modelSkipGram)


# Learning Edge Embedding
model = rw.learnEdgeEmbedding(modelSkipGram)

# Plot Embedding
ge.plot_2DEmbedding(rw)

# Save embedding to disk
ge.saveEmbedding(data_dir, dataset, rw)

node1 = 35
node2 = 40
# Get Embedding for a node
emb = rw.getNodeEmbedding(node1)
print("Node Embedding", emb)
#
# Get Embedding for an edge
emb = rw.getEdgeEmbedding(node1, node2)
print("Edge Embedding", emb)
```