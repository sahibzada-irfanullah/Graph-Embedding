# import module
import ge

# Set Path to data
dataset = "cora"
data_dir = "../data"

# input graph
myGraph = ge.loadGraph(data_dir, dataset)

# Set Parameters
embedDim = 10 # embedding size
numbOfWalksPerVertex = 2 # walks per vertex
walkLength = 5 # walk lenght
lr =0.25 # learning rate
windowSize = 3 # window size

# # Set Parameters
# embedDim = 128 # embedding size
# numbOfWalksPerVertex = 10 # walks per vertex
# walkLength = 80 # walk lenght
# lr =0.025 # learning rate
# windowSize = 10 # window size




# DeepWalk
rw = ge.DeepWalk(myGraph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
                 windowSize=windowSize, lr = lr)

# Skip Gram model
model_skip_gram = ge.SkipGramModel(rw.totalNodes, rw.embedDim)

print("Learning Embedding")
# Learning Node Embedding
model = rw.learnNodeEmbedding(model_skip_gram)

# Learning Edge Embedding
# model = dw.learnEdgeEmbedding(model_skip_gram)

# Plot Embedding
ge.plot_2DEmbedding(rw)

# Save embedding to disk
print("Saving embedding to disk")
ge.saveEmbedding(data_dir, dataset, rw)

print("Generating embedding for a node and edge")
node1 = 35
node2 = 40
# Get Embedding for a node
emb = rw.getNodeEmbedding(node1)
print("Node Embedding", emb.data)
#
# # Get Embedding for an edge
emb = rw.getEdgeEmbedding(node1, node2)
print("Edge Embedding", emb.data)



