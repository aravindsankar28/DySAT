from subprocess import call
import sys
import argparse

from subprocess import Popen
import os, json, time

"""NOTE: This script includes nearly all tf flag parameters as input arguments, which feed as input 
through a generated config file."""

parser = argparse.ArgumentParser(description='Run script parameters')

# Script specific parameters -> min and max time steps for executing different train files.
# Time step range [min_time, max_time to train different models (both included).
# Min time step is always 2 since we require at least one snapshot each for train and test.

parser.add_argument('--min_time', type=int, nargs='?', default=2, help='min_time step')

parser.add_argument('--max_time', type=int, nargs='?', default=16, help='max_time step')

# NOTE: Ensure that the execution is split into different ranges so that GPU memory errors are avoided.
# IncSAT must be executed sequentially

parser.add_argument('--run_parallel', type=str, nargs='?', default='False',
                    help='By default, sequential execution of different time steps (Note: IncSAT must be sequential)')

# Necessary parameters for log creation.
parser.add_argument('--base_model', type=str, nargs='?', default='DySAT',
                    help='Base model (DySAT/IncSAT)')

# Additional model string to save different parameter variations.
parser.add_argument('--model', type=str, nargs='?', default='default',
                    help='Additional model string')

# Experimental settings.
parser.add_argument('--dataset', type=str, nargs='?', default='Enron_new',
                    help='dataset name')

parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                    help='GPU_ID (0/1 etc.)')

parser.add_argument('--epochs', type=int, nargs='?', default=200,
                    help='# epochs')

parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                    help='Validation frequency (in epochs)')

parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                    help='Testing frequency (in epochs)')

parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                    help='Batch size (# nodes)')

# 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
parser.add_argument('--featureless', type=str, nargs='?', default='True',
                    help='True if one-hot encoding.')

parser.add_argument('--max_gradient_norm', type=float, nargs='?', default=1.0,
                    help='Clip gradients to this norm')

# Tunable hyper-params

# TODO: Implementation has not been verified, performance may not be good.
parser.add_argument('--use_residual', type=str, nargs='?', default='False',
                    help='Use residual')

# Number of negative samples per positive pair.
parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                    help='# negative samples per positive')

# Walk length for random walk sampling.
parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                    help='Walk length for random walk sampling')

# Weight for negative samples in the binary cross-entropy loss function.
parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                    help='Weightage for negative samples')

parser.add_argument('--learning_rate', type=float, nargs='?', default=0.001,
                    help='Initial learning rate for self-attention model.')

parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                    help='Spatial (structural) attention Dropout (1 - keep probability).')

parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                    help='Temporal attention Dropout (1 - keep probability).')

parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                    help='Initial learning rate for self-attention model.')

# Architecture params

parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                    help='Encoder layer config: # attention heads in each GAT layer')

parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                    help='Encoder layer config: # units in each GAT layer')

parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                    help='Encoder layer config: # attention heads in each Temporal layer')

parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                    help='Encoder layer config: # units in each Temporal layer')

parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                    help='Position wise feedforward')

parser.add_argument('--window', type=int, nargs='?', default=-1,
                    help='Window for temporal attention (default : -1 => full)')

args = parser.parse_args()

min_time = int(args.min_time)
max_time = int(args.max_time)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_input():
    yes = {'yes', 'y', 'ye'}
    no = {'no', 'n'}

    while True:
        choice = raw_input("Enter your choice (yes/no) (yes => continue without any changes, no => exit) : ").lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' \n")


print (args)
output_dir = "./logs/" + args.base_model + "_" + args.model

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

with open(output_dir + '/flags_{}.json'.format(args.dataset), 'w') as outfile:
    json.dump(vars(args), outfile)

with open(output_dir + '/flags_{}.txt'.format(args.dataset), 'w') as outfile:
    for k, v in vars(args).items():
        outfile.write("{}\t{}\n".format(k, v))

# Dump args to flags file.

train_file = "train.py" if args.base_model == "DySAT" else "train_incremental.py"

# Here, t=2 => learn on graph (idx = 0) and predict the links of graph (idx = 1).
commands = []
for t in range(args.min_time, args.max_time + 1):
    commands.append(' '.join(
        ["python", train_file, "--time_steps", str(t), "--base_model", args.base_model, "--model", args.model,
         "--dataset", args.dataset]))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


print ("Args parallel param: ", args.run_parallel)

if str2bool(args.run_parallel) and args.base_model == 'DySAT':
    print ("Running time steps {} to {} in parallel on GPU {}".format(args.min_time, args.max_time, args.GPU_ID))
    processes = []
    for cmd in commands:
        pid = Popen(cmd, shell=True)
        time.sleep(10)
        processes.append(pid)

    for p in processes:
        p.wait()
else:
    print ("Running time steps {} to {} sequentially on GPU {}".format(args.min_time, args.max_time, args.GPU_ID))
    time.sleep(1)
    for cmd in commands:
        print ("Call ", cmd)
        call(cmd, shell=True)
