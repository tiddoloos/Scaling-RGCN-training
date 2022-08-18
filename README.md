# Scaling Relational Graph Convolutional Network training with Graph Summaries and Entity Embedding transfer

This repositor contains the implmentation to scale Relational Graph Convolutional Network (R-GCN) trianing with graph summaries, proposed in this thesis.

### Abstract
Relational Graph Convolutional Network (R-GCN) training on real-world graphs is challenging. Storing gradient information during R-GCN training on real-world graphs, exceeds available memory on most single devices.
Recent work demonstrated to scale R-GCN training with a summary graph. The appropriate graph summarization technique is of- ten unknown and graph and task dependent.
Overcoming this problem, we propose R-GCN pre-training on multiple graph summaries, produced with attribute and (k)-forward bisimulation summarization techniques.
With pre-training on graph summaries, multiple entity embeddings and one set R-GCN weights can be obtained.
We applied Summation, Multi- Layer Perceptron and Multi-Head Attention models to transfer multi- ple entity embeddings and R-GCN weights to a new R-GCN model.
With the new R-GCN model we conducted full-graph training for entity type prediction.
Our contribution to existing research is three-fold, as this work demonstrated how: graph summaries reduce parameters for R-GCN training, while maintaining or improving R-GCN performance;
the creation of graph summaries can be included in R-GCN training to maintain or improve R-GCN performance, while reducing computational time;
graph summaries in combination with Multi-Layer Percep- tron and Multi-Head Attention can be applied to scale R-GCN training and maintain or improve R-GCN performance, while freezing the gradients of the R-GCN weights after summary graph pre-training.

![model pipelines](https://github.com/tiddoloos/Scaling-RGCN-training/blob/main/paper/pipelines.jpg?raw=true)

### Requirements
To use the repository we recommend creating a virtual environment, e.g. with conda. Use requirements.txt to install the dependencies:
```
conda create -n scaling_rgcn python=3.8 
conda activate scaling_rgcn
pip install -r requirements.txt
```