"""
graph_models.py
Main class implementation file for Janossy GraphSAGE
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import logging
import random
from collections import defaultdict
from math import factorial
from janossy_gs.data_loaders import *
import numpy as np
import time
import sys
from sklearn.metrics import f1_score
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc, loss_type):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.loss_type = loss_type
        self.fc_final = nn.Linear(enc.embed_dim, num_classes, bias=False)
        self.fc_final.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.fc_final.weight)

    def forward(self, nodes):

        if self.enc.aggregator_class == "MeanAggregator":
            embeds = self.enc(nodes)
            scores = self.fc_final.weight.mm(embeds)
            return scores.t()

        elif self.enc.aggregator_class == "LSTMAggregator":
            embeds = self.enc(nodes)
            scores = self.fc_final(embeds)
            return scores


    def loss_and_performance(self, nodes, labels):

        scores = self.forward(nodes)  # The forked code uses forward, but one could instead use the call method inherited from nn.Module.

        if self.loss_type == "cross_entropy":
            #
            # Get loss, track grads
            # Targets are long tensors
            #
            loss_fn = nn.CrossEntropyLoss()
            torch_labels = torch.LongTensor(labels).to(device)
            loss_value = loss_fn(scores, torch_labels.squeeze())
            #
            #  Get another metric of performance, don't track grads
            #
            with torch.no_grad():
                np_preds = scores.data.cpu().numpy().argmax(axis=1)
                performance = f1_score(labels.squeeze(), np_preds, average="micro")

        elif self.loss_type == "binary_cross_entropy_with_logits":
            #
            # Get loss, track grads
            # Expects input to be an arbitrary real number
            # Targets to be {0.0, 1.0}, i.e. float tensors.
            #
            loss_fn = torch.nn.BCEWithLogitsLoss()
            torch_labels = torch.FloatTensor(labels).to(device)
            loss_value = loss_fn(scores, torch_labels.squeeze())
            #
            #  Get another metric of performance, don't track grads
            #
            with torch.no_grad():
                np_probs = torch.sigmoid(scores).data.cpu().numpy()
                preds = np.where(np_probs > 0.5, 1, 0)
                performance = f1_score(labels.squeeze(), preds, average="micro")
        else:
            raise NotImplementedError("This loss is not recognized: {}".format(self.loss_type))

        return loss_value, performance


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=25, base_model=None):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.embed_dim = embed_dim

        if base_model is not None:
            self.base_model = base_model

        self.aggregator_class = self.aggregator.__class__.__name__

        if self.aggregator_class == "MeanAggregator":
            self.weight = nn.Parameter(torch.FloatTensor(embed_dim, 2 * self.feat_dim))
            init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """

        if self.aggregator_class == "LSTMAggregator":
            # For LSTM do this for nodes individually
            combined = torch.zeros([1, self.embed_dim]).to(device)
            for node in nodes:
                node_emb = self.aggregator.forward(node, self.adj_lists[int(node)], self.num_sample)  # The forked code uses forward, but one could instead use the call method inherited from nn.Module.
                combined = torch.cat([combined, node_emb], dim=0)

            combined = combined[1:, :]
        #
        #
        #
        elif self.aggregator_class == "MeanAggregator":
            #TODO: Some runtime errors on GPU, but works fast enough on a cpu
            neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
            self_feats = self.features(torch.LongTensor(nodes).to(device))

            combined = torch.cat([self_feats, neigh_feats], dim=1)
            combined = F.relu(self.weight.mm(combined.t()))

        else:
            raise RuntimeError("Aggregator not recognized {}".format(self.aggregator_class))

        return combined


class LSTMAggregator(nn.Module):
    """
    Aggregates a node's embeddings using Seq of neighbors' embeddings
    """

    def __init__(self, features, shape, embedding_dim, activate, normalize, lstm_count=1, permute=False):
        """
        Initializes the aggregator for a specific graph.
        :argument features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        :argument shape -- input dimension of the vectors in the sequence fed into LSTM
        :argument embedding_dim -- Output dim of the LSTM.
        :argument activate -- Boolean to apply nonlinear activation
        :argument normalize -- Boolean to apply normalization
        :argument lstm_count -- Number of random permutations (and thus number of times lstm is forwarded)
                                one can experiment with whether this impacts the results, but we did not in our work.

        The following flags are deprecated carry-overs from previous implementations.
        :argument permute --- whether to perform random permutation
        """
        assert embedding_dim % 2 == 0 and isinstance(embedding_dim, int), "embedding dim should be an even positive integer"
        assert isinstance(activate, bool), "`activate` must be a boolean"
        assert isinstance(normalize, bool), "`normalize` must be a boolean"

        if permute:
            raise Warning("The use of boolean flag `permute' is deprecated in this implementation and does nothing.")

        super(LSTMAggregator, self).__init__()
        self.features = features
        self.permute = permute
        self.embedding_dim = embedding_dim
        self.half_embedding_dim = embedding_dim/2  # For readability

        self.lstm = nn.LSTM(shape, embedding_dim, batch_first=True)
        self.fc_neib = nn.Linear(embedding_dim, self.half_embedding_dim, bias=False)
        self.fc_neib.weight = nn.Parameter(torch.FloatTensor(self.half_embedding_dim, embedding_dim))
        self.fc_x = nn.Linear(shape, self.half_embedding_dim, bias=False)
        self.fc_x.weight = nn.Parameter(torch.FloatTensor(self.half_embedding_dim, shape))
        init.xavier_uniform_(self.fc_x.weight)
        init.xavier_uniform_(self.fc_neib.weight)

        self.activate = activate
        self.normalize = normalize
        self.lstm_count = lstm_count

    def forward(self, node, to_neighs, num_sample=25):
        """
        nodes --- list of nodes in a batch (or one node in case of LSTM)
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.

        Note:
        Here we follow the logic of https://github.com/williamleif/graphsage-simple/blob/master/graphsage/
        (The TensorFlow implementation, not the reference PyTorch implementation which does not have an LSTM aggregator)
        This code deviates slightly from what is written in the graphsage paper.

        For example:
        In the GraphSAGE TensorFlow code,
        Layer ONE: Apply ReLU but not normalization
        Layer TWO: Apply Normalization but not ReLU.
        """
        #
        # Sample neighbors
        #
        _sample = random.sample  # Local pointers to functions (speed hack)
        _lstm_count = self.lstm_count  # Number of lstms to use for this node (we use fewer if num_nodes < num_lstm)

        if num_sample is not None and len(to_neighs) >= num_sample:
            unique_nodes_list = _sample(to_neighs, num_sample)
        else:
            # No sampling is performed
            unique_nodes_list = list(to_neighs)
        #
        # Make `_lstm_count` random permutations of the neighbor set
        # >>  First check that _lstm_count is not larger than the possible number of permutations
        if factorial(len(unique_nodes_list)) < _lstm_count:
            _lstm_count = factorial(len(unique_nodes_list))

        shuffled_lists = self.shuffle_helper(unique_nodes_list, _lstm_count)
        #
        # Get features in neighborhood from lookup.
        #
        neib_feats = defaultdict()

        for shuflist_num in range(_lstm_count):
            neib_feats[shuflist_num] = self.features(torch.LongTensor(shuffled_lists[shuflist_num]).to(device))
        #
        # Aggregate neighbors using each LSTM, average results
        #
        x = self.features(torch.LongTensor([node]).to(device))

        agg_neib = defaultdict()
        agg_neib_mean = torch.zeros([1, self.embedding_dim], requires_grad=True).to(device)
        for shuflist_num in neib_feats:
            # As long as x only represents one node, x.size(0) == 1
            # So we re-shape to (1, number neighbors, feature (or embedding) dim).
            agg_neib[shuflist_num] = neib_feats[shuflist_num].view(x.size(0), -1, neib_feats[shuflist_num].size(1))
            agg_neib[shuflist_num], _ = self.lstm(agg_neib[shuflist_num])
            agg_neib[shuflist_num] = agg_neib[shuflist_num][:, -1, :]  # Final state of LSTM
            agg_neib_mean = torch.add(agg_neib_mean, agg_neib[shuflist_num])

        agg_neib_mean = agg_neib_mean/float(_lstm_count)
        #
        # Apply a fully connected neural network to the aggregated features and to the current node's vertex feature.
        # Then: CONCATENATE
        # Following logic of https://github.com/williamleif/GraphSAGE
        #
        x_emb = self.fc_x(x)

        neib_emb = self.fc_neib(agg_neib_mean)

        to_feats = torch.cat([x_emb, neib_emb], dim=1)
        #
        # Activations and Normalization
        if self.activate:
            to_feats = F.relu(to_feats)

        if self.normalize:
            to_feats = F.normalize(to_feats, p=2, dim=1)

        return to_feats

    def shuffle_helper(self, list_shuffle, count):
        shuffled = []
        for x in range(count):
            shuffled.append(random.sample(list_shuffle, k=len(list_shuffle)))
        return shuffled


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features):
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        _sample = random.sample
        if num_sample is not None:
            samp_neighs = [_set(_sample(to_neigh, num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = torch.autograd.Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mask.to(device)
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        embed_matrix = self.features(torch.LongTensor(unique_nodes_list).to(device))
        to_feats = mask.mm(embed_matrix)
        return to_feats


class JanossyGraphSage:
    """
    Runs GraphSAGE using Janossy pooling using k-ary or n-ary/piSGD models
    For a given dataset and a given set of hyperparameters
    """
    def __init__(self, **job_args):
        # Read command line arguments into a dictionary
        self.job_args = job_args

        # Dictionary whose keys="dataset string", values="appropriate function to load data"
        self.dataset_loader = dict()
        self.dataset_loader["cora"] = load_cora
        self.dataset_loader["pubmed"] = load_pubmed
        self.dataset_loader["citation"] = load_citation
        self.dataset_loader["ppi"] = load_ppi
        self.dataset_loader["reddit"] = load_reddit

        # Initialize a dictionary of the ``training data`` we would like to save.
        # i.e.: Loss at each batch, test set performance, training time, etc.
        self.train_info = {"train_loss": [], "val_loss": [], "batch_times": [],
                           "train_F1": [], "val_F1": [], "test_F1": [], "test_probs": []
                           }

    def load_data_from_name(self, dataset):
        """
        Calls the appropriate function to load the graph data
        Makes test/train/val splits

        :argument dataset: a string describing the dataset to be loaded
        """
        assert isinstance(dataset, str)
        dataset = dataset.lower()

        if dataset in self.dataset_loader:
            loader = self.dataset_loader[dataset]
            self.feat_data, self.labels, self.adj_lists, self.num_nodes, self.num_feats, self.num_classes = loader(self.job_args["input_dir"])
            #
            # Check the expected format for adj list
            if not (isinstance(self.adj_lists, defaultdict) and self.adj_lists.default_factory is set):
                raise NotImplementedError("""Adjacency lists are currently expected to be defaultdicts of sets.
                                          \nUsing sets avoids checking for duplicates during each forward pass.""")

        else:
            raise NotImplementedError("No loader implemented for {}.".format(dataset))
        #
        # Do test/train split
        #
        # >> These datasets have pre-defined splits (ppi has "training graphs" and "testing graphs"):
        if dataset == "ppi":
            split_dict = np.load(self.job_args['input_dir'] + "ppi_data_split_indices.npy")
            self.train_idx = list(split_dict.item().get('train_ids'))
            self.test_idx = list(split_dict.item().get('test_ids'))
            self.val_idx = list(split_dict.item().get('val_ids'))
        # These do not have pre-defined splits and are just randomly permuted here
        else:
            rand_indices = np.random.permutation(self.num_nodes)
            num_test = self.job_args["num_test"]
            num_val = self.job_args["num_val"]

            self.test_idx = rand_indices[:num_test]
            self.val_idx = rand_indices[num_test:(num_test + num_val)]
            self.train_idx = list(rand_indices[(num_test + num_val):])

    # Assuming that we either perform exact k-ary inference
    # or approximate n-ary inference
    def build_kary_model(self):
        self.model_type = "kary"
        logging.info("Pytorch device type: {}".format(device))

        # Create a lookup table from nodes to their features.
        features = nn.Embedding(self.num_nodes, self.num_feats)
        features.weight = nn.Parameter(torch.FloatTensor(self.feat_data), requires_grad=False)
        #
        # Build up pooling layers: Assuming 2 pooling layers (2 graph conv layers) in this implementation.
        #
        agg1 = MeanAggregator(features)
        agg1 = agg1.to(device)
        enc1 = Encoder(features, self.num_feats, self.job_args['embedding_dim_one'], self.adj_lists, agg1)
        enc1 = enc1.to(device)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t())
        agg2 = agg2.to(device)
        enc2 = Encoder(       lambda nodes: enc1(nodes).t(), self.job_args['embedding_dim_one'], self.job_args['embedding_dim_two'], self.adj_lists, agg2, base_model=enc1)
        enc2 = enc2.to(device)
        #
        # Set number of neighbors to sample
        #
        enc1.num_sample = self.job_args["num_samples_one"]
        enc2.num_sample = self.job_args["num_samples_two"]
        #
        # Build graphsage object
        #
        self.graphsage = SupervisedGraphSage(self.num_classes, enc2, self.job_args['loss_type'])
        self.graphsage = self.graphsage.to(device)

    # TODO: Better name for this function, in light of recent conversations.
    def build_nary_model(self):
        """
        Create a Pytorch model for the n-ary Janossy model (uses the full sequence)
        Inference will be done via pi SGD
        """
        self.model_type = "nary"
        logging.info("Pytorch device type: {}".format(device))

        # Create a lookup table from nodes to their features.
        features = nn.Embedding(self.num_nodes, self.num_feats)
        features.weight = nn.Parameter(torch.FloatTensor(self.feat_data), requires_grad=False)
        #
        # Build up pooling layers: Assuming 2 pooling layers (2 graph conv layers) in this implementation.
        #
        agg1 = LSTMAggregator(features, self.num_feats, self.job_args['embedding_dim_one'], activate=True, normalize=False, lstm_count=self.job_args['num_lstms'])
        agg1 = agg1.to(device)
        enc1 = Encoder(       features, self.num_feats, self.job_args['embedding_dim_one'], self.adj_lists, agg1)
        enc1 = enc1.to(device)

        agg2 = LSTMAggregator(lambda nodes: enc1(nodes), self.job_args['embedding_dim_one'], self.job_args['embedding_dim_two'], activate=False, normalize=True, lstm_count=self.job_args['num_lstms'])
        agg2 = agg2.to(device)
        enc2 = Encoder(       lambda nodes: enc1(nodes), self.job_args['embedding_dim_one'], self.job_args['embedding_dim_two'], self.adj_lists, agg2, base_model=enc1)
        enc2 = enc2.to(device)
        #
        # Set number of neighbors to sample
        #
        enc1.num_sample = self.job_args["num_samples_one"]
        enc2.num_sample = self.job_args["num_samples_two"]
        #
        # Build graphsage object
        #
        self.graphsage = SupervisedGraphSage(self.num_classes, enc2, self.job_args['loss_type'])
        self.graphsage = self.graphsage.to(device)

    def _train_work(self, optimizer, batch, batch_nodes):
        """
        This helper function runs optimization, inside of a loop.
        The loop can run over epochs in a traditional fashion or over just batches.
        """
        #
        # Compute training loss and update parameters
        #
        optimizer.zero_grad()
        start_time = time.time()
        loss, performance = self.graphsage.loss_and_performance(batch_nodes, self.labels[np.array(batch_nodes)])
        loss.backward()
        optimizer.step()
        end_time = time.time()
        #
        # Validation performance
        #
        # (For debugging)
        # random.seed(batch)
        # np.random.seed(batch)

        with torch.no_grad():
            val_loss, val_performance = self.graphsage.loss_and_performance(self.val_idx,
                                                                            self.labels[np.array(self.val_idx)])
        #
        # Log it
        #
        logging.info("=" * 50)
        logging.info(
            "Batch: %3d | Train Loss: %.5f | Val Loss: %.5f | Train F1 : %.5f" % (batch, loss, val_loss, performance))
        #
        # Save info to dict
        #
        self.train_info['train_loss'].append(loss.item())
        self.train_info['val_loss'].append(val_loss.item())
        self.train_info['train_F1'].append(performance)
        self.train_info['val_F1'].append(val_performance)
        self.train_info['batch_times'].append(end_time - start_time)

    def train_model(self):
        """
        Wrapper around training logic
        Will either use traditional epoch strategy or the inductive batch strategy of the forked code,
        depending on the dataset.
        """
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, self.graphsage.parameters()), lr=self.job_args['lr'])
        logging.info("Training:\n")

        if not self.job_args['typical_epochs']:
            # Don't use the typical epoch approach, per forked code.
            # Rather than looping over epochs, we just choose several mini-batches.
            for batch in range(self.job_args['num_batches']):
                # (Setting the same seed for training and validation can be useful for debugging)
                # (For example, calculating validation loss induces more calls to `random`, changing
                # the permutations at train-time.  In other words, this optimization will give
                # different results depending upon whether validation-set performance is calculated.
                # One can argue that seeds should not be called at each batch, however)
                # random.seed(batch)
                # np.random.seed(batch)
                # Allocate a batch.
                random.shuffle(self.train_idx)
                batch_nodes = self.train_idx[:self.job_args['batch_size']]
                self._train_work(optimizer, batch, batch_nodes)
        else:
            # Traditional epoch approach
            # random.seed(batch)
            # np.random.seed(batch)
            logging.info("Using traditional epoch approach.\n")

            batch = 0  # For printing the current batch of training.
            batch_size = self.job_args['batch_size']
            batches_per_epoch = int(len(self.train_idx)/batch_size)

            for epoch in range(self.job_args['num_epochs']):
                logging.info("*~"*25)
                logging.info("Epoch {}".format(epoch))
                logging.info("*~"*25)

                random.shuffle(self.train_idx)

                for this_batch in range(batches_per_epoch):

                    batch_start = this_batch * batch_size
                    batch_stop = this_batch * batch_size + batch_size
                    batch_nodes = self.train_idx[batch_start:batch_stop]
                    self._train_work(optimizer, batch, batch_nodes)
                    batch += 1

                # Save a checkpoint because these take so long
                logging.info("Saving a checkpoint....")
                checkpoint_path = self.job_args['weights_path'][0:-4]
                checkpoint_path += "cpt" + str(epoch+1) + ".pth"
                torch.save(self.graphsage.state_dict(), checkpoint_path)

    def test_model(self, evaluate_multiple_perms=False, num_perms=None):
        """
        Compute test-set performance using the F1 score.
        At inference time, we sample multiple permutations to compute the average output
        We will achieve this by simply calling `forward` multiple times.
        Also, since all we need is the argmax of these probabilities, we will simply sum (not average)

        :argument: evaluate_multiple_perms: Boolean, are we running
        to obtain performance scores with multiple inference-time permutations?
        This will typically be done after training and saving weights.

        > May not yield equivalent performance as during training -- better or worse --
           due to random shuffling.
        """
        loss_type = self.job_args['loss_type']
        test_output = torch.zeros([len(self.test_idx), self.num_classes], dtype=torch.float).to(device)
        labels = self.labels[np.array(self.test_idx)]

        if evaluate_multiple_perms:
            num_permutations = num_perms
            out_dict = dict()
        else:
            num_permutations = self.job_args["inf_permute_number"]

        with torch.no_grad():
            for index in range(num_permutations):
                logging.info("Inference iteration: {}".format(index))
                # Forward (automatically uses different permutations)
                scores = self.graphsage.forward(self.test_idx)

                if loss_type == "cross_entropy":
                    prob = F.softmax(scores, dim=1)
                elif loss_type == "binary_cross_entropy_with_logits":
                    prob = torch.sigmoid(scores)

                # self.train_info["test_probs"].append(prob)
                test_output += prob

                if evaluate_multiple_perms or index == num_permutations-1:
                    if loss_type == "cross_entropy":
                        test_preds = test_output.data.cpu().numpy().argmax(axis=1)
                    elif loss_type == "binary_cross_entropy_with_logits":
                        test_mean = (test_output/float(index + 1)).data.cpu().numpy()
                        test_preds = np.where(test_mean > 0.5, 1, 0)

                    performance = f1_score(labels.squeeze(), test_preds, average="micro")
                    logging.info("\nTest set F1: %.5f" % performance)

                    if evaluate_multiple_perms:
                        out_dict[index] = performance

        if evaluate_multiple_perms:
            return out_dict
        else:
            self.train_info["test_F1"] = performance

    def save_info(self):
        filename = self.job_args['train_data_path']
        pickle.dump(self.train_info, open(filename, "wb"))
        logging.info("Training data saved: " + filename)

