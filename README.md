# Scaling Relational Graph Convolutional Network training with Graph Summaries and Entity Embedding transfer

This repositor contains the implmentation to scale Relational Graph Convolutional Network (R-GCN) trianing with graph summaries, proposed in my thesis.
By training R-GCN on graph summaries entity embeddings are learned.
Training on multiple graph summaries yield multiple entity embeddings for entities in the summary graphs
The e

![alt text](https://github.com/tiddoloos/Scaling-RGCN-training/blob/main/pdfs.jpg?raw=true)

### Requirements
We recommend creating a virtual environment, e.g. with conda. Use requirements.txt to install the dependencies:
```
conda create -n scaling_rgcn python=3.8 
conda activate scaling_rgcn
pip install -r requirements.txt
```
