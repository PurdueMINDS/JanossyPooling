###########################################################################################
# Ryan Murphy
# Janossy GraphSAGE
# preprocess_ppi.py
#
# Separate preprocessing from loading during train/val/test...
# so we don't perform redundant preprocessing when running training w/ different random inits
###########################################################################################

import numpy as np
from networkx.readwrite import json_graph, write_gpickle
import json
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def preprocess_ppi(data_dir):
    """
    Avoid some redundant calculation when working w/ ppi dataset.
    We will also benefit from the speed of loading numpy data.
    Some minimal processing is still left to train/test time for consistency w/ other datasets.
    :param data_dir: Directory of ppi data
    :return: Nothing, but saves processed data to disk
    train/test/val indices
    features
    graph
    labels
    """
    if data_dir[-1] != "/":
        data_dir += "/"

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(data_dir + "ppi-G.json")))
    feats = np.load(data_dir + "ppi-feats.npy")
    class_map = json.load(open(data_dir + "ppi-class_map.json"))  # Targets.
    id_map = json.load(open(data_dir + "ppi-id_map.json"))
    #
    # Create Test/train/val splits
    #
    print("Creating test/train/val splits")
    conversion = lambda my_node: int(my_node)
    id_map = {conversion(k): int(v) for k, v in id_map.items()}

    data_splits = dict()
    data_splits['train_ids'] = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
    data_splits['test_ids'] = np.array([n for n in G.nodes() if G.node[n]['test']])
    data_splits['val_ids'] = np.array([n for n in G.nodes() if G.node[n]['val']])

    np.save(data_dir + "ppi_data_split_indices.npy", data_splits)
    #
    # Scale the feature data
    #
    print("Processing the feature data...")
    train_feats = feats[data_splits['train_ids']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    np.save(data_dir + "preprocessed_ppi__features.npy", feats)
    #
    # Labels AKA targets
    #
    print("Processing labels...")
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda val: val
    else:
        lab_conversion = lambda val: int(val)
    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}
    ids = [n for n in G.nodes()]
    labels = np.array([class_map[i] for i in ids])

    np.save(data_dir + "preprocessed_ppi_labels.npy", labels)
    #
    # Process the graph
    #
    # > Remove all nodes that do not have val/test annotations
    broken_count = 0
    for node in G.nodes():
        if 'val' not in G.node[node] or 'test' not in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(
        broken_count))
    #
    # > Make sure the graph has edge train_removed annotations
    # > (some datasets might already have this..)
    print("Cleaning edges...")
    for edge in G.edges():
        if G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or G.node[edge[0]]['test'] or G.node[edge[1]]['test']:
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    # Save graph
    print("Saving graph...")
    write_gpickle(G, data_dir + "preprocessed_ppi_graph.pkl")


if __name__ == '__main__':
    preprocess_ppi("/scratch/brown/murph213/ppi/")
