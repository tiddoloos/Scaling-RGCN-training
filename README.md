# Scaling Relational Graph Convolutional Network Training with Graph Summaries and Entity Embedding transfer

This repository contains the implementation to scale Relational Graph Convolutional Network (R-GCN) training with graph summaries, proposed in the thesis [Scaling R-GCN training with Graph Summaries and Entity Embedding Transfer](https://github.com/tiddoloos/Scaling-RGCN-training/blob/main/thesis/Scaling_RGCN_Training_with_Graph_Summaries_and_Entity_Embedding_Transfer_Tiddo_Loos_2574974.pdf).

## Abstract
Relational Graph Convolutional Network (R-GCN) training on real-world graphs is challenging. Storing gradient information during R-GCN training on real-world graphs, exceeds available memory on most single devices.
Recent work demonstrated to scale R-GCN training with a summary graph. The appropriate graph summarization technique is often unknown and graph and task dependent.
Overcoming this problem, we propose R-GCN pre-training on multiple graph summaries, produced with attribute and (k)-forward bisimulation summarization techniques.
With pre-training on graph summaries, multiple entity embeddings and one set R-GCN weights can be obtained.
We applied Summation, MultiLayer Perceptron and Multi-Head Attention models to transfer multiple entity embeddings and R-GCN weights to a new R-GCN model.
With the new R-GCN model we conducted full-graph training for entity type prediction.
Our contribution to existing research is three-fold, as this work demonstrated how: graph summaries reduce parameters for R-GCN training, while maintaining or improving R-GCN performance;
the creation of graph summaries can be included in R-GCN training to maintain or improve R-GCN performance, while reducing computational time;
graph summaries in combination with Multi-Layer Perceptron and Multi-Head Attention can be applied to scale R-GCN training and maintain or improve R-GCN performance, while freezing the gradients of the R-GCN weights after summary graph pre-training.

![model pipelines](https://github.com/tiddoloos/Scaling-RGCN-training/blob/main/thesis/pipelines.jpg?raw=true)

## Requirements
To use the repository, we recommend creating a virtual environment, e.g. with conda.
Use requirements.txt to install the dependencies:
```
conda create -n scaling_rgcn python=3.8 
conda activate scaling_rgcn
pip install -r requirements.txt
```
The AM dataset is too large push to github.
Download the AM dataset, including graph summaries, [here](https://drive.google.com/uc?id=1r9bA0B75dvdlwEHBgpfOOhoRIpCZdHTr&export=download).
Unpack AM.zip and add like `./graphs/AM`.

The `./graphs` folder contains graphs datasets. Each graph folder, e.g.`AM`, contains attribute summaries (`attr`) and (k)-forward bisimulaiton summaries (`bisim`).
The attribute graph summaries are stored in `./graphs/{dataset}/attr/sum`.
The (k)-forward bisimulaiton summaries are stored in `./graphs/{dataset}/bisim/sum`
For each graph atrribute (and (k)-forward bisimulation) summary there exists a map file in `./graphs/{dataset}/attr/map`.
`./graphs/{dataset}/one/` contains a single summary graph (either attribute or (k)-forward bisimulation) which must be added munually.

## Experiments
We provide example commands to reproduce our experiments.
The commands be should run from the root directory of the repository.
The aim is to scale R-GCN training for entity type prediction.
We display examples for running the experiments on the AIFB dataset (`-dataset AIFB`) for 5 iterations (`-i 5`).
By default the experiments run for 51 epochs.
There are three different models to choose from: `summation`, `mlp` and `attention`.
For a detailed description of the models we refer to the thesis (section 5.2).

#### Multiple Summary Graphs
The following command runs the experiment where pre-training on the summary graphs, present in the `.graphs/AIFB/attr` folder, occurs.
After training on graph summaries, the embeddings and R-GCN weights are transferred to a new R-GCN model.
Then, full original graph training is carried out for entity type prediction.
The program will automatically save results to `./results`.
Also, the results are plotted automatically.
```
python main.py -dataset AIFB -sum attr -i 5 -exp attention
```
#### Single Summary Graph
To run the single summary graph experiment, copy the desired graph summary to the `./graphs/AIFB/one/sum`.
Copy its complementing map graph to `./graphs/AIFB/one/map`.
```
python main.py -dataset AIFB -sum one -i 5 -exp attention
```
#### Embedding and R-GCN Weights Transfer
It can be decided to transfer either the entity embeddings or the R-GCN weights from summary graph training with the following commands:
```
python main.py -sum attr -i 5 -exp attention -w_trans False -e_trans True
python main.py -sum attr -i 5 -exp attention -w_trans True -e_trans False
```
When entity embeddings are not transferred from summary graph pre-training, the entity embedding for training on the full original graph gets newly initialized.
#### Freezing Embedding and R-GCN Weights
The gradients of the R-GCN weights can be frozen after transferring them from the summary graph model with `-w_grad`.
Note that the `mlp` and `attention` model contain layer spicific weights and the `summation` does not.
We recommend to use the following command with the `mlp` and `attention` model only:
```
python main.py -sum attr -i 5 -exp attention -w_grad False 
```
The entity embedding can be frozen or unfrozen  by setting `-e_freeze` to `True` or `False`.
By default, the transferred embedding is frozen: `-e_freeze True`.


## Create Summary Graphs
With the following command the incoming, outgoing and the incoming/outgoing attribute summary graphs can be created for a graph dataset:
```
python graphs/createAttributeSum.py -dataset AIFB
```
For the creation of (k)-forward bisimulation summary graphs we refer to [FLUID](https://github.com/t-blume/fluid-framework).
