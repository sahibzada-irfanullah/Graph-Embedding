# Graph Embedding
Generating KHop in the MapReduce Infrastructure.
# Requirement
- Python  3.7.5


## How to utilize a code? (Tutorial)

#### input graph
```
inputGraph = Your input graph goes here
```

#### Configurable Parameter for Graph Embedding
```
embedDim = 2 # embedding size
numbOfWalksPerVertex = 2 # walks per vertex
walkLength = 4 # walk lenght
lr =0.025 # learning rate
windowSize = 3 # window size
```


#### Graph Embedding Model
```
# DeepWalk
dw = DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)
# Skip Gram model
modelSkipGram = SkipGramModel(dw.totalNodes, dw.embedDim)

# Learning Node Embedding
model = dw.learnNodeEmbedding(modelSkipGram)
```


#### Get Embedding
```
nodeID = 10
dw.getNodeEmbedding(nodeID)
```

