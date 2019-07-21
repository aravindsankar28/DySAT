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
from models.IncSAT.models import IncSAT
from utils.preprocess import *
from utils.utilities import *
from utils.incremental_minibatch import *


np.random.seed(123)
tf.set_random_seed(123)

flags = tf.app.flags
FLAGS = flags.FLAGS

# Assumes as input -> proper base model and model name to get the folder to load the flags from parser.
output_dir = "./logs/{}_{}/".format(FLAGS.base_model, FLAGS.model)
config_file = output_dir + "flags_{}.json".format(FLAGS.dataset)

with open(config_file, 'r') as f:
    config = json.load(f)
    for name, value in config.items():
        if name in FLAGS.__flags:
            FLAGS.__flags[name].value = value

print("Updated flag params", map(lambda x: (x[0], x[1].value), FLAGS.__flags.items()))

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

logging.info(map(lambda flag: (flag[0], flag[1].value), FLAGS.__flags.items()))

# Create file name for result log csv from certain flag parameters.
output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

# Adj matrix at time t -> Should include nodes at (t+1) too, so that embeddings can be learnt.
# For baselines, the full matrix should be provided -- with all nodes till say - (t+1).
num_time_steps = FLAGS.time_steps
graphs, adjs = load_graphs(FLAGS.dataset)
if FLAGS.featureless:
    feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
else:
    feats = load_feats(FLAGS.dataset)

num_time_steps = FLAGS.time_steps
context_pairs = get_context_pairs_incremental(graphs[num_time_steps - 2])
train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
    get_evaluation_data(adjs, num_time_steps, FLAGS.dataset)

print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))
logging.info("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

num_time_steps = FLAGS.time_steps  # NOTE: minimum value of num_time_steps is 2
assert num_time_steps < len(adjs) + 1  # So that, (t+1) can be predicted.

# Construct training data - create pairs only for the last time step => idx = time_steps - 2.

# Create the adj_train so that it includes nodes from (t+1) but only edges from t.
new_G = nx.MultiGraph()
new_G.add_nodes_from(graphs[num_time_steps - 1].nodes(data=True))

for e in graphs[num_time_steps - 2].edges():
    new_G.add_edge(e[0], e[1])

graphs[num_time_steps - 1] = new_G
adjs[num_time_steps - 1] = nx.adjacency_matrix(new_G)
graph_train = graphs[num_time_steps - 2]
adj_train = nx.adjacency_matrix(graph_train)
adj_train = normalize_graph_gcn(adj_train)
num_features = feats[0].shape[1]
feat_train = preprocess_features(feats[num_time_steps - 2])[1]
num_features_nonzero = feat_train[1].shape[0]


def construct_placeholders():
    # Define placeholders
    placeholders = {
        'node_1': tf.placeholder(tf.int32, shape=(None,), name="node_1"),  # [None,1] for each time step.
        'node_2': tf.placeholder(tf.int32, shape=(None,), name="node_2"),  # [None,1] for each time step.
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),  # [None,1]
        'prev_hidden_embeds': [tf.placeholder(tf.float32, shape=(1, None, None)) for t in range(0, num_time_steps - 2)],
        'feature': tf.sparse_placeholder(tf.float32, shape=(None, num_features), name="feat"),
        'adj': tf.sparse_placeholder(tf.float32, shape=(None, None), name="adj"),
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
    }
    return placeholders

prev_hidden_embeds = []
if num_time_steps > 2:  # in case of 2, nothing to be done.
    # Embed - saving and loading follow same convention as eval files.
    try:
        print("Trying to load from file with path -> ",
              "{}/{}_{}_hidden_embeds.npz".format(MODEL_DIR, FLAGS.dataset, str(num_time_steps - 3)))
        prev_hidden_embeds = \
            np.load("{}/{}_{}_hidden_embeds.npz".format(MODEL_DIR, FLAGS.dataset, str(num_time_steps - 3)),
                    encoding='bytes')['data']
    except IOError:
        raise ValueError("Cannot load previous step(s) hidden layer embeddings")

print("Initializing session")
# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
placeholders = construct_placeholders()
minibatchIterator = IncrementalNodeMinibatchIterator(graph_train, feat_train, adj_train, prev_hidden_embeds,
                                                     placeholders, batch_size=FLAGS.batch_size,
                                                     context_pairs=context_pairs)

model = IncSAT(placeholders, num_features, num_features_nonzero, minibatchIterator.degs)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# Result accumulators.
epochs_test_result = defaultdict(lambda: [])
epochs_val_result = defaultdict(lambda: [])
epochs_embeddings = []
epochs_attn_wts_means = []
epochs_attn_wts_vars = []

for epoch in range(FLAGS.epochs):
    minibatchIterator.shuffle()
    epoch_loss = 0.0
    it = 0
    print('Epoch: %04d' % (epoch + 1))
    while not minibatchIterator.end():
        # Construct feed dictionary
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        feed_dict.update({placeholders['spatial_drop']: FLAGS.spatial_drop})
        feed_dict.update({placeholders['temporal_drop']: FLAGS.temporal_drop})
        t = time.time()
        # Training step
        _, train_cost, current_cost, reg_cost = sess.run([model.opt_op, model.loss, model.graph_loss, model.reg_loss],
                                                         feed_dict=feed_dict)
        # Print results
        logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, train_cost))
        logging.info("Mini batch Iter: {} current_loss= {:.5f}".format(it, current_cost))
        logging.info("Mini batch Iter: {} reg_loss= {:.5f}".format(it, reg_cost))
        epoch_loss += train_cost
        it += 1

    if epoch % FLAGS.test_freq == 0:
        minibatchIterator.test_reset()
        feed_dict.update({placeholders['spatial_drop']: 0.0})
        feed_dict.update({placeholders['temporal_drop']: 0.0})
        emb = sess.run(model.final_output_embeddings, feed_dict=feed_dict)[:, FLAGS.time_steps - 2, :]
        emb = np.array(emb)
        val_results, test_results, _, _ = evaluate_classifier(train_edges, train_edges_false, val_edges,
                                                              val_edges_false, test_edges, test_edges_false, emb, emb)

        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]

        if (epoch == 0) or (epoch > 0 and epoch_auc_val >= max(epochs_val_result["HAD"])):
            save_path = MODEL_DIR + "/" + "model_{}_{}.ckpt".format(FLAGS.dataset, FLAGS.time_steps - 2)
            saver.save(sess, save_path)
            print("Saving model at epoch {}".format(epoch))
            logging.info("Saving model at epoch {}".format(epoch))
            hidden_embeds = sess.run(model.hidden_embeds, feed_dict=feed_dict)
            np.savez("{}/{}_{}_hidden_embeds.npz".format(MODEL_DIR, FLAGS.dataset, str(num_time_steps - 2)),
                     data=hidden_embeds)

        print("Epoch {}, Val AUC {}".format(epoch, epoch_auc_val))
        print("Epoch {}, Test AUC {}".format(epoch, epoch_auc_test))
        logging.info("Val results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_val))
        logging.info("Test results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_test))

        epochs_test_result["HAD"].append(epoch_auc_test)
        epochs_val_result["HAD"].append(epoch_auc_val)
        epochs_embeddings.append(emb)

    epoch_loss /= it
    print("Mean Loss at epoch {} : {}".format(epoch, epoch_loss))

# Result log for link prediction.
best_epoch = epochs_val_result["HAD"].index(max(epochs_val_result["HAD"], key=lambda feat: feat[0]))

print("Best epoch ", best_epoch)
logging.info("Best epoch {}".format(best_epoch))

val_results, test_results, _, _ = evaluate_classifier(graphs[FLAGS.time_steps - 1], train_edges, train_edges_false,
                                                      val_edges, val_edges_false, test_edges, test_edges_false,
                                                      epochs_embeddings[best_epoch], epochs_embeddings[best_epoch])

print("Best epoch val results {}\n".format(val_results))
print("Best epoch test results {}\n".format(test_results))

logging.info("Best epoch val results {}\n".format(val_results))
logging.info("Best epoch test results {}\n".format(test_results))

write_to_csv(val_results, output_file, FLAGS.model, FLAGS.dataset, num_time_steps, mod='val')
write_to_csv(test_results, output_file, FLAGS.model, FLAGS.dataset, num_time_steps, mod='test')

# Save final embeddings in the save directory.
emb = epochs_embeddings[best_epoch]
np.savez(SAVE_DIR + '/{}_embs_{}_{}.npz'.format(FLAGS.model, FLAGS.dataset, FLAGS.time_steps - 2), data=emb)
