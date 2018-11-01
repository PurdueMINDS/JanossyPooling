# Janossy Pooling
### Authors: Ryan L. Murphy and Balasubramaniam Srinivasan

## Overview:
This is the code for Janossy Pooling: Learning Deep Permutation-Invariant Functions for Variable-Size Inputs

It mainly has two tasks, arthimetic tasks on sequences of integers and graph vertex classification
There are five different arthimetic tasks, namely sum, range, unique sum, unique count and variance
Vertex classification is performed on three different graphs

## Requirements
* pytorch 0.4.0 or later - which can be downloaded [here](www.pytorch.org)
* python 2.7

## How to Run
For the sequence based tasks, please use the following format:
* python train.py -m "model name" -t "task" -l "no of hidden layers in rho" -lr "learning rate" -b "batch size"
* Permitted models are {deepsets,janossy_nary,lstm,gru}
* Permitted tasks are {sum,range,unique_sum,unique_count,variance}

For the graph based tasks, we have provided an example below:
* python train_janossy_gs.py --model_type=nary --dataset=cora    --num_samples_one=5  --num_samples_two=5  --inf_permute_number=20 --seed=1 --embedding_dim_one=256 --embedding_dim_two=256 --num_test=1000 --num_val=500 --lr=0.005 --num_batches=100 --batch_size=256 --output_dir=<> --input_dir=<>

## Data
* For the arthimetic tasks, the data is generated on the fly. You can adjust the number of training, test and validation examples used.
* For the graph tasks, the following datasets were used:
  - [PPI](https://snap.stanford.edu/graphsage/ppi.zip)
  - Cora
  - Pubmed


## Citation
If you use this code, please cite our paper
```


```
