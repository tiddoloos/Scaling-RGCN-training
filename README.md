# Scaling Relational Graph Convolutional Network training with Graph Summaries and Entity Embedding transfer

This repositor contains the implmentation to scale Relational Graph Convolutional Network (R-GCN) trianing with graph summaries, proposed in this thesis.

### Abstract
Relational Graph Convolution Network (R-GCN) training on real-world graphs is challenging.
Storing gradient information during R-GCN training on real-world graphs, exceeds available memory on most single devices.
Recent work demonstrated the possibility to scale R-GCN training with a summary graph.
The appropriate graph summarization technique is graph and task dependent and often unknown.
Overcoming this problem, we propose R-GCN pre-training on multiple graph summaries, produced with attribute and (k)-forward bisimulation summarization techniques.
With pre-training on graph summaries, multiple entity embeddings and one set R-GCN weights can be obtained.
We applied \textit{Summation}, \textit{Multi-Layer Perceptron} and \textit{Multi-Head Attention} models to transfer multiple entity embeddings and R-GCN weights to a new R-GCN model.
With the new R-GCN model we conducted full-graph training for entity type prediction.
Our contribution to existing research is three-fold, as this work demonstrated how:
graph summaries can scale R-GCN training and maintain or improve R-GCN performance, while reducing trainable parameters; 
the creation of graph summaries can be included in R-GCN training to scale R-GCN training and maintain or improve R-GCN performance, while reducing computational time;
graph summaries in combination with \textit{Multi-Layer Perceptron} and \textit{Multi-Head Attention} can be applied to scale R-GCN training and maintain or improve R-GCN performance, while freezing R-GCN weights after summary graph pre-training.

![model pipelines](https://github.com/tiddoloos/Scaling-RGCN-training/blob/main/paper/pipelines.png?raw=true)

### Requirements
We recommend creating a virtual environment, e.g. with conda. Use requirements.txt to install the dependencies:
```
conda create -n scaling_rgcn python=3.8 
conda activate scaling_rgcn
pip install -r requirements.txt
```