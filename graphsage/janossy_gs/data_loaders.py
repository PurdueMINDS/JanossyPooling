"""
data_loaders.py

Functions to read in graph datasets.
"""
import numpy as np
from collections import defaultdict
from networkx.readwrite import read_gpickle
import json
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def load_cora(indir):
    # hardcoded for simplicity...
    num_nodes = 2708
    num_feats = 1433
    num_classes = 7
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(indir + "cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = map(float, info[1:-1])  # assumes python 2.7
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
    #
    adj_lists = defaultdict(set)
    with open(indir + "cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    #
    return feat_data, labels, adj_lists, num_nodes, num_feats, num_classes


def load_pubmed(indir):
    # hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    num_classes = 3
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open(indir + "Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i-1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    #
    adj_lists = defaultdict(set)
    with open(indir + "Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    #
        return feat_data, labels, adj_lists, num_nodes, num_feats, num_classes


def load_citation(indir):
    raise NotImplementedError
    return feat_data, labels, adj_lists, num_nodes, num_feats, num_classes


def load_ppi(indir):
    # Hard-coded for simplicity.
    num_classes = 121
    #
    G = read_gpickle(indir + "preprocessed_ppi_graph.pkl")
    feat_data = np.load(indir + "preprocessed_ppi__features.npy")
    labels = np.load(indir + "preprocessed_ppi_labels.npy")
    training_set_size = len(np.load(indir + "ppi_data_split_indices.npy").item().get('train_ids'))
    assert training_set_size == 44906, "Original code assumed training set size 44906, which has changed. Please review"
    #
    num_nodes = feat_data.shape[0]
    num_feats = feat_data.shape[1]
    #
    adj_lists = defaultdict(set)
    for node in G.nodes():
        temp = dict(G[node])
        if int(node) < training_set_size:
            adj_lists[node] = set([x for x in temp if temp[x]['train_removed'] is False])
        else:
            adj_lists[node] = set([x for x in temp])
    #
    return feat_data, labels, adj_lists, num_nodes, num_feats, num_classes


def load_reddit(indir):
    raise NotImplementedError
    return feat_data, labels, adj_lists, num_nodes, num_feats, num_classes

