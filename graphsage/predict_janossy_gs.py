"""
Balasubramaniam Srinivasan and Ryan L Murphy
Load saved weights and perform testing
We may use this to study the impact of the number of inference-time permutations.
(Did not do so during training to avoid adding extra overhead)
"""
from train_janossy_gs import *
import pickle
import os.path

NUM_INF_PERMS = 21  # Store permutation performance up through this number


def make_inference_paths(log_path_, inf_permute_number):
    """Convert the training-time log to a inference-time log and results dictionary"""
    # Remove directory, leave just file name
    log_path_= log_path_.split("/")[-1]
    assert log_path_[0:3] == "log"
    # Strip out "permute number" information and remove first three letters ("log")
    base_file = log_path_.replace("_inf_permute_number_" + str(inf_permute_number), "")[3:]
    # Strip out ".log", because this is a base for all file paths
    base_file = base_file.replace(".log", "")
    # Add "inflog", inference log and the range of new values to the beginning
    new_log = "inflog_1_thru_" + str(NUM_INF_PERMS) + base_file + ".log"
    #
    new_dat = "infdat_1_thru_" + str(NUM_INF_PERMS) + base_file + ".pkl"
    #
    return new_log, new_dat

if __name__ == "__main__":
    # We can use the same command used to launch the jobs for training.
    # At training, we chose one inf_permute_number for test-set performance,
    # which is now in the file name.
    # So, we will replace it here after loading.
    args = parse_args()
    args_dict = vars(args)
    outdir = args_dict['output_dir']

    assert args.model_type != 'kary', "Only makes sense to run this for permutation sensitive models."

    weights_path, log_path, train_data_path = build_out_path(args_dict)
    if not os.path.isfile(weights_path):
        print("Weights file does not exist")
        print(weights_path)
        sys.exit(0)  # This is not necessarily a problem.  I will just run prediction over a set of indices, skip if file doesn't exist

    inf_log_path, inf_dat_path = make_inference_paths(log_path, args.inf_permute_number)

    args_dict['train_data_path'] = train_data_path
    # check if this pickle file exists

    set_logger(outdir  + inf_log_path)
    logging.info("Executed on: {}".format(datetime.datetime.now()))
    logging.info("\nCommand-line inputs:\n")
    logging.info(args_dict)
    logging.info("\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #
    # BUILD MODEL OBJECT SO WE CAN LOAD WEIGHTS INTO IT
    #
    computation_class = JanossyGraphSage(**args_dict)

    logging.info("Loading data.....")  # We use same test set, which is constructed here.
    computation_class.load_data_from_name(args.dataset)

    logging.info("Building model.....")
    computation_class.build_nary_model()
    #
    # Load model
    #
    logging.info("\nLoading trained model.....")
    computation_class.graphsage.load_state_dict(torch.load(weights_path))

    #
    # Forward
    #
    F1_dict = computation_class.test_model(evaluate_multiple_perms=True, num_perms=NUM_INF_PERMS)
    #
    # Save
    #
    pickle.dump(F1_dict, open(outdir + inf_dat_path, 'wb'))
    logging.info("Performance saved to {}".format(inf_dat_path))

