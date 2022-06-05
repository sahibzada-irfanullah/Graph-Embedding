# Graph Embedding
Generating KHop in the MapReduce Infrastructure.
# Requirement
- Python  3.7.5


# How to utilize a code? (Tutorial)
### input graph
```
my_graph = nx.from_pandas_edgelist(edgelist, edge_attr="label")
```

Where
- Khop: required khop number
- path/inputgraph: hdfs path to the input graph
- path/khop: hdfs path to the output

# Output:
After running the code successfully, all the generated khops will be stored in "/khop" on hdfs as:

NodeID_KHop.txt

Provided for academic use only
