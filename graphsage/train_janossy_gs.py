"""
Balasubramaniam Srinivasan and Ryan L Murphy
Train GraphSAGE with Janossy pooling

Use Janossy pooling in the GraphSAGE task with k-ary and piSGD.

"""

import torch
import numpy as np
import argparse
import random
from collections import OrderedDict
from janossy_gs.graph_models import JanossyGraphSage
import sys
import os
import datetime
import pickle

sys.path.append("..")
from training_utils import *



def parse_args():
    """
    Input hyperparameters and other command-line arguments to control training & testing
    :return: args
    """
    parser = argparse.ArgumentParser()

    # Assuming we either do k-ary exact or n-ary piSGD
    parser.add_argument('-m', '--model_type', required=True, type=str, help='Model type: kary or nary')
    parser.add_argument('-od', '--output_dir', required=True, type=str, help='Output directory')
    parser.add_argument('-p', '--inf_permute_number', required=True, type=int, help='Number of permutations at inference time')
    parser.add_argument('-ds', '--dataset', required=True, type=str, help='Graph Dataset')
    parser.add_argument('-id', '--input_dir', required=True, type=str, help='Input data directory')
    parser.add_argument('-sd', '--seed', required=True, type=int, help='A seed value.  Set to different values when using different random inits')
    parser.add_argument('-edo', '--embedding_dim_one', required=True, type=int, help='Dimension of Encoder embedding for first layer  (set nonpositive to use full seq) ')  # assuming 2 layers
    parser.add_argument('-edt', '--embedding_dim_two', required=True, type=int, help='Dimension of Encoder embedding for second layer (set nonpositive to use full seq) ')  # assuming 2 layers
    parser.add_argument('-nso', '--num_samples_one', required=True, type=int, help='Number of neighbors to sample at layer 1')
    parser.add_argument('-nst', '--num_samples_two', required=True, type=int, help='Number of neighbors to sample at layer 2')
    parser.add_argument('-lr',  '--lr', required=True, type=float, help='Learning rate (step size) for Adam optimizer')
    parser.add_argument('-bs', '--batch_size', required=True, type=int, help='Size of mini-batch')
    parser.add_argument('-nte', '--num_test', required=False, type=int,    help='Number nodes to use for testing')
    parser.add_argument('-nva', '--num_val', required=False, type=int, help='Number nodes to use for validation')
    parser.add_argument('-nb',  '--num_batches', required=False, type=int, help='Number of mini-batches to run')
    #
    parser.add_argument('-nl', '--num_lstms', required=False, default=1, type=int, help='Number of LSTMS, ie number of train time permutations (defaults to 1)')
    parser.add_argument('-lot', '--loss_type', required=False, default='cross_entropy', type=str, help='Type of loss function to use')
    # Do epochs?
    # GraphSAGE does not use typical epoch approach with cora, pubmed (for example), but they do for ppi
    parser.add_argument('--typical_epochs', action='store_true', help="Use this flag to optimize using the typical epoch approach.")
    parser.add_argument('-ne', '--num_epochs', required=False, default=-1, type=int, help="Number of epochs if doing traditional epoch-based training")

    # Parse and clean arguments
    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    args.model_type = args.model_type.lower()

    if args.output_dir[-1] != "/":
        args.output_dir += "/"

    if args.input_dir[-1] != "/":
        args.input_dir += "/"

    assert args.dataset in ("cora", "pubmed", "ppi", "citation", "reddit"), "No implementation for entered dataset: ``{}''".format(args.dataset)
    assert args.model_type in ("kary", "nary"), "Model type should be `kary' or `nary'"
    assert args.embedding_dim_one % 2 == 0, "embedding_dim_one  must be an even positive integer"
    assert args.embedding_dim_two % 2 == 0, "embedding_dim_two  must be an even positive integer"
    assert os.path.isdir(args.output_dir), "output directory does not exist.\n" + args.output_dir
    assert os.path.isdir(args.input_dir), "input directory does not exist."
    assert args.loss_type in ("cross_entropy", "binary_cross_entropy_with_logits")

    assert (args.num_samples_one < 1 and args.num_samples_two < 1) or \
           (args.num_samples_one > 0 and args.num_samples_two > 1), "num samples 1, 2 must both be pos or nonpositive"

    if args.num_samples_one < 1 and args.num_samples_two < 1:
        args.num_samples_one = None  # Indicates that we don't sample -- full seq used.
        args.num_samples_two = None

    if args.typical_epochs:
        assert args.num_epochs > 0, "If --typical_epochs is activated, --num_epochs must be a positive integer"
    else:
        assert args.num_batches is not None and args.num_batches > 0

    if args.dataset in ("cora", "pubmed"):
        assert (args.num_test is not None) and (args.num_val is not None), "These datasets require you to specify num test and num val"
        assert args.num_test > 0 and args.num_val > 0, "Please enter a positive number of test/val samples."


    return args


def build_out_path(d):
    """
    Build the path to where we will save the model, log, and training data.
    :param d: Dictionary command line args (all the cmd-line args get used to build the filepath)
    :return: path to weights_path, log_path, train_data_path
    """
    assert "output_dir" in d, "Dictionary must have output dir"

    path = d.pop("output_dir")

    weights_path = path + "weights_"
    log_path = path + "log_"
    train_data_path = path + "train_data_"

    # Sort dictionary for consistency
    odict = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

    for key, val in zip(odict.keys(), odict.values()):
        if key not in ("input_dir", "loss_type", "typical_epochs", "num_epochs"):
            text = str(key) + "_" + str(val) + "_"
            weights_path += text
            log_path += text
            train_data_path += text

    weights_path += ".pth"
    log_path += ".log"
    train_data_path += ".pkl"

    return weights_path, log_path, train_data_path


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)

    weights_path, log_path, train_data_path = build_out_path(args_dict)
    args_dict['train_data_path'] = train_data_path

    args_dict['weights_path'] = weights_path

    set_logger(log_path)

    logging.info("Executed on: {}".format(datetime.datetime.now()))
    logging.info("\nCommand-line inputs:\n")
    logging.info(args_dict)
    logging.info("\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #
    # Initialize object to hold model and data
    #
    computation_class = JanossyGraphSage(**args_dict)
    #
    # Inject data into computation_class
    # (dimensions of data are used to specify model: must load data first)
    #
    logging.info("Loading data.....")
    computation_class.load_data_from_name(args.dataset)
    #
    # Build model
    #
    logging.info("Building model.....")
    if args.model_type == "nary":
        computation_class.build_nary_model()
    elif args.model_type == "kary":
        computation_class.build_kary_model()
    #
    # Train model
    #
    computation_class.train_model()
    logging.info("\nSaving trained model.....")
    torch.save(computation_class.graphsage.state_dict(), weights_path)
    #
    # Summarize the different model components and their shapes
    #
    logging.info("\n\nModel architecture summary:")
    for kk in computation_class.graphsage.state_dict().keys():
        logging.info(kk)
        logging.info(computation_class.graphsage.state_dict()[kk].shape)

    model_parameters = filter(lambda p: p.requires_grad, computation_class.graphsage.parameters())
    print(map(lambda arg : type(arg), model_parameters))
    logging.info("Total number of training params: {:,}".format(get_n_params(computation_class.graphsage)))
    logging.info("\n" + "-"*50)
    #
    # Test Model.
    #
    #
    logging.info("\nTesting.......")
    computation_class.test_model()
    computation_class.save_info()

