from skipgram import SkipGramModel
from utils import plot_2DEmbedding
from node2vec import Node2vec
from struc2vec import Struc2Vec
from deepWalk import DeepWalk
from utils import saveEmbedding, loadGraph

# Set Path to data
# dataset = "cora"
dataset = "cora - Copy.cites"
data_dir = "../data"


# input graph
my_graph = loadGraph(data_dir, dataset)

# Set Parameters (For fast completion we kept the parameters simple)
embedDim = 2 # embedding size
numbOfWalksPerVertex = 2 # walks per vertex
walkLength = 4 # walk lenght
lr =0.25 # learning rate
windowSize = 3 # window size


# DeepWalk
dw = DeepWalk(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)

# # Node2Vec
# dw = Node2vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
#               windowSize=windowSize, lr=lr, p = 0.5, q = 0.8)
# #
# # Struc2Vec
# dw = Struc2Vec(my_graph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
#                windowSize=windowSize, lr = lr, stay_prob=0.3)


# Skip Gram model
model_skip_gram = SkipGramModel(dw.totalNodes, dw.embedDim)

print("Learning Embedding")
# Learning Node Embedding
model = dw.learnNodeEmbedding(model_skip_gram)

# Learning Edge Embedding
# model = dw.learnEdgeEmbedding(model_skip_gram)

# Plot Embedding
plot_2DEmbedding(dw)

# Save embedding to disk
print("Saving embedding to disk")
saveEmbedding(data_dir, dataset, dw)

print("Generating embedding for a node and edge")
node1 = 35
node2 = 40
# Get Embedding for a node
emb = dw.getNodeEmbedding(node1)
print("Node Embedding", emb.data)
#
# # Get Embedding for an edge
emb = dw.getEdgeEmbedding(node1, node2)
print("Edge Embedding", emb.data)



