# Janossy Pooling
### Authors: Ryan L. Murphy and Balasubramaniam Srinivasan

## Overview:
This is the code for [Janossy Pooling: Learning Deep Permutation-Invariant Functions for Variable-Size Inputs](https://arxiv.org/abs/1811.01900).

We evaluate different Janossy Pooling models for tasks similar to those found in [Deep Sets](https://github.com/manzilzaheer/DeepSets) and [GraphSAGE](https://github.com/williamleif/GraphSAGE).
Our implementation follows these, as well as the reference [PyTorch implementation of GraphSAGE](https://github.com/williamleif/graphsage-simple/). The latter repo also contains two datasets that we use.

The first set of tasks is to perform arithmetic on sequences of integers: sum, range, unique sum, unique count, and variance.  Note that these functions are all permutation-invariant (symmetric).

The second set of tasks learns vertex embeddings in graphs for vertex classification.  The data are described below.

Please see the supplementary section for a brief description and summary of the code. 

## Requirements
* PyTorch 0.4.0 or later - which can be downloaded [here](https://www.pytorch.org)
* Python 2.7

## How to Run
For the sequence based tasks, please use the following format:
* python train.py -m "model name" -t "task" -l "number of hidden layers in rho" -lr "learning rate" -b "batch size"
* Permitted models are {deepsets,janossy_nary,lstm,gru}
* Permitted tasks are {sum,range,unique_sum,unique_count,variance}

For the graph based tasks, we have provided an example below:
* python train_janossy_gs.py --model_type=nary --dataset=cora    --num_samples_one=5  --num_samples_two=5  --inf_permute_number=20 --seed=1 --embedding_dim_one=256 --embedding_dim_two=256 --num_test=1000 --num_val=500 --lr=0.005 --num_batches=100 --batch_size=256 --output_dir=<> --input_dir=<>

We recommend training these models on a GPU.

## Data
* For the arithmetic tasks, the data is generated on the fly as described in our paper. You can adjust the number of training, test and validation examples used.
* For the graph tasks, the following datasets were used:
  - [PPI](https://snap.stanford.edu/graphsage/ppi.zip)
  - Cora, [available at the GraphSAGE PyTorch repo](https://github.com/williamleif/graphsage-simple/)
  - Pubmed, [available at the GraphSAGE PyTorch repo](https://github.com/williamleif/graphsage-simple/)

Cora and Pubmed are described in Sen et al., 2008, and PPI in Zitnik and Leskovec, 2017.  They are described in Hamilton 2017.
Please see [our paper](https://arxiv.org/abs/1811.01900) for further details, including our test/train/validation splits. 

## Questions
Please feel free to reach out to Ryan L. Murphy (murph213  at  purdue.edu) if you have any questions.

## Citation
If you use this code, please consider citing:
```
@inproceedings{
murphy2018janossy,
title={Janossy Pooling: Learning Deep Permutation-Invariant Functions for Variable-Size Inputs},
author={Ryan L. Murphy and Balasubramaniam Srinivasan and Vinayak Rao and Bruno Ribeiro},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=BJluy2RcFm},
}
```
