from __future__ import division
from __future__ import print_function

import json
import os
import time
from datetime import datetime

import logging
import scipy

from eval.link_prediction import evaluate_classifier, write_to_csv
from flags import *
from models.DySAT.models import DySAT
from utils.minibatch import *
from utils.preprocess import *
from utils.utilities import *

np.random.seed(123)
tf.set_random_seed(123)

flags = tf.app.flags
FLAGS = flags.FLAGS

# Assumes a saved base model as input and model name to get the right directory.
output_dir = "./logs/{}_{}/".format(FLAGS.base_model, FLAGS.model)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

config_file = output_dir + "flags_{}.json".format(FLAGS.dataset)

with open(config_file, 'r') as f:
    config = json.load(f)
    for name, value in config.items():
        if name in FLAGS.__flags:
            FLAGS.__flags[name].value = value

print("Updated flags", FLAGS.flag_values_dict().items())

# Set paths of sub-directories.
LOG_DIR = output_dir + FLAGS.log_dir
SAVE_DIR = output_dir + FLAGS.save_dir
CSV_DIR = output_dir + FLAGS.csv_dir
MODEL_DIR = output_dir + FLAGS.model_dir

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if not os.path.isdir(CSV_DIR):
    os.mkdir(CSV_DIR)

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU_ID)

datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()

# Setup logging
log_file = LOG_DIR + '/%s_%s_%s_%s_%s.log' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day), str(FLAGS.time_steps))

log_level = logging.INFO
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

logging.info(FLAGS.flag_values_dict().items())

# Create file name for result log csv from certain flag parameters.
output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

# model_dir is not used in this code for saving.

# utils folder: utils.py, random_walk.py, minibatch.py
# models folder: layers.py, models.py
# main folder: train.py
# eval folder: link_prediction.py

"""
#1: Train logging format: Create a new log directory for each run (if log_dir is provided as input). 
Inside it,  a file named <>.log will be created for each time step. The default name of the directory is "log" and the 
contents of the <>.log will get appended per day => one log file per day.

#2: Model save format: The model is saved inside model_dir. 

#3: Output save format: Create a new output directory for each run (if save_dir name is provided) with embeddings at 
each 
time step. By default, a directory named "output" is created.

#4: Result logging format: A csv file will be created at csv_dir and the contents of the file will get over-written 
as per each day => new log file for each day.
"""

# Load graphs and features.

num_time_steps = FLAGS.time_steps

graphs, adjs = load_graphs(FLAGS.dataset)
if FLAGS.featureless:
    feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
else:
    feats = load_feats(FLAGS.dataset)

num_features = feats[0].shape[1]
assert num_time_steps < len(adjs) + 1  # So that, (t+1) can be predicted.

adj_train = []
feats_train = []
num_features_nonzero = []
loaded_pairs = False

# Load training context pairs (or compute them if necessary)
context_pairs_train = get_context_pairs(graphs, num_time_steps)

# Load evaluation data.
train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
    get_evaluation_data(adjs, num_time_steps, FLAGS.dataset)

# Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
# inductive testing.
new_G = nx.MultiGraph()
new_G.add_nodes_from(graphs[num_time_steps - 1].nodes(data=True))

for e in graphs[num_time_steps - 2].edges():
    new_G.add_edge(e[0], e[1])

graphs[num_time_steps - 1] = new_G
adjs[num_time_steps - 1] = nx.adjacency_matrix(new_G)

print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))
logging.info("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

# Normalize and convert adj. to sparse tuple format (to provide as input via SparseTensor)
adj_train = map(lambda adj: normalize_graph_gcn(adj), adjs)

if FLAGS.featureless:  # Use 1-hot matrix in case of featureless.
    feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in feats if
             x.shape[0] <= feats[num_time_steps - 1].shape[0]]
num_features = feats[0].shape[1]

feats_train = map(lambda feat: preprocess_features(feat)[1], feats)
num_features_nonzero = [x[1].shape[0] for x in feats_train]

def construct_placeholders(num_time_steps):
    min_t = 0
    if FLAGS.window > 0:
        min_t = max(num_time_steps - FLAGS.window - 1, 0)
    placeholders = {
        'node_1': [tf.placeholder(tf.int32, shape=(None,), name="node_1") for _ in range(min_t, num_time_steps)],
        # [None,1] for each time step.
        'node_2': [tf.placeholder(tf.int32, shape=(None,), name="node_2") for _ in range(min_t, num_time_steps)],
        # [None,1] for each time step.
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),  # [None,1]
        'features': [tf.sparse_placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                     range(min_t, num_time_steps)],
        'adjs': [tf.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for i in
                 range(min_t, num_time_steps)],
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='spatial_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='temporal_drop')
    }
    return placeholders


print("Initializing session")
# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

placeholders = construct_placeholders(num_time_steps)

minibatchIterator = NodeMinibatchIterator(graphs, feats_train, adj_train,
                                          placeholders, num_time_steps, batch_size=FLAGS.batch_size,
                                          context_pairs=context_pairs_train)
print("# training batches per epoch", minibatchIterator.num_training_batches())

model = DySAT(placeholders, num_features, num_features_nonzero, minibatchIterator.degs)
sess.run(tf.global_variables_initializer())

# Result accumulator variables.
epochs_test_result = defaultdict(lambda: [])
epochs_val_result = defaultdict(lambda: [])
epochs_embeds = []
epochs_attn_wts_all = []

for epoch in range(FLAGS.epochs):
    minibatchIterator.shuffle()
    epoch_loss = 0.0
    it = 0
    print('Epoch: %04d' % (epoch + 1))
    epoch_time = 0.0
    while not minibatchIterator.end():
        # Construct feed dictionary
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        feed_dict.update({placeholders['spatial_drop']: FLAGS.spatial_drop})
        feed_dict.update({placeholders['temporal_drop']: FLAGS.temporal_drop})
        t = time.time()
        # Training step
        _, train_cost, graph_cost, reg_cost = sess.run([model.opt_op, model.loss, model.graph_loss, model.reg_loss],
                                                       feed_dict=feed_dict)
        epoch_time += time.time() - t
        # Print results
        logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, train_cost))
        logging.info("Mini batch Iter: {} graph_loss= {:.5f}".format(it, graph_cost))
        logging.info("Mini batch Iter: {} reg_loss= {:.5f}".format(it, reg_cost))
        logging.info("Time for Mini batch : {}".format(time.time() - t))

        epoch_loss += train_cost
        it += 1

    print("Time for epoch ", epoch_time)
    logging.info("Time for epoch : {}".format(epoch_time))
    if (epoch + 1) % FLAGS.test_freq == 0:
        minibatchIterator.test_reset()
        emb = []
        feed_dict.update({placeholders['spatial_drop']: 0.0})
        feed_dict.update({placeholders['temporal_drop']: 0.0})
        if FLAGS.window < 0:
            assert FLAGS.time_steps == model.final_output_embeddings.get_shape()[1]
        emb = sess.run(model.final_output_embeddings, feed_dict=feed_dict)[:,
              model.final_output_embeddings.get_shape()[1] - 2, :]
        emb = np.array(emb)
        # Use external classifier to get validation and test results.
        val_results, test_results, _, _ = evaluate_classifier(train_edges,
                                                              train_edges_false, val_edges, val_edges_false, test_edges,
                                                              test_edges_false, emb, emb)

        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]

        print("Epoch {}, Val AUC {}".format(epoch, epoch_auc_val))
        print("Epoch {}, Test AUC {}".format(epoch, epoch_auc_test))
        logging.info("Val results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_val))
        logging.info("Test results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_test))

        epochs_test_result["HAD"].append(epoch_auc_test)
        epochs_val_result["HAD"].append(epoch_auc_val)
        epochs_embeds.append(emb)
    epoch_loss /= it
    print("Mean Loss at epoch {} : {}".format(epoch, epoch_loss))

# Choose best model by validation set performance.
best_epoch = epochs_val_result["HAD"].index(max(epochs_val_result["HAD"]))

print("Best epoch ", best_epoch)
logging.info("Best epoch {}".format(best_epoch))

val_results, test_results, _, _ = evaluate_classifier(train_edges, train_edges_false, val_edges, val_edges_false,
                                                      test_edges, test_edges_false, epochs_embeds[best_epoch],
                                                      epochs_embeds[best_epoch])

print("Best epoch val results {}\n".format(val_results))
print("Best epoch test results {}\n".format(test_results))

logging.info("Best epoch val results {}\n".format(val_results))
logging.info("Best epoch test results {}\n".format(test_results))

write_to_csv(val_results, output_file, FLAGS.model, FLAGS.dataset, num_time_steps, mod='val')
write_to_csv(test_results, output_file, FLAGS.model, FLAGS.dataset, num_time_steps, mod='test')

# Save final embeddings in the save directory.
emb = epochs_embeds[best_epoch]
np.savez(SAVE_DIR + '/{}_embs_{}_{}.npz'.format(FLAGS.model, FLAGS.dataset, FLAGS.time_steps - 2), data=emb)
