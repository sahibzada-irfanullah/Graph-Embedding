# Graph Embedding
### Introduction
This module provides the services and implementation for various
family of graph embedding algorithms.


### How to utilize a code? (Tutorial)
#### input graph
```
# inputGraph = Your input graph goes here
# For example,
# Set Path to data
dataset = "cora"
dataset = "cora - Copy"
data_dir = "../cora"

# Load Data
data_dir = os.path.expanduser(data_dir)
edgelist = pd.read_csv(os.path.join(data_dir, dataset + ".cites"), sep='\t', header=None, names=["target", "source"])

# input graph
my_graph = nx.from_pandas_edgelist(edgelist)
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
dw = DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)
              
              
# Node2Vec
dw = Node2vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
               windowSize=windowSize, lr=lr, p = 0.5, q = 0.8)

# Struc2Vec
dw = Struc2Vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)
              
# Skip Gram model
modelSkipGram = SkipGramModel(dw.totalNodes, dw.embedDim)

# Choose whether want Node embedding or edge embedding
# Learning Node Embedding
model = dw.learnNodeEmbedding(modelSkipGram)


# Learning Edge Embedding
model = dw.learnEdgeEmbedding(model_skip_gram)

# Plot Embedding
plot_2DEmbedding(dw)

# Save embedding to disc
saveEmbedding(data_dir, dataset, dw)

node1 = 35
node2 = 40
# Get Embedding for a node
emb = dw.getNodeEmbedding(node1)
print("Node Embedding", emb)
#
# Get Embedding for an edge
emb = dw.getEdgeEmbedding(node1, node2)
print("Edge Embedding", emb)
```
