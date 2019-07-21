from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class IncrementalNodeMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes to sample context pairs for a batch of nodes.

    graphs -- list of networkx graphs
    features -- list of (scipy) sparse node attribute matrices
    adjs -- list of adj matrices (of the graphs)
    placeholders -- standard tensorflow placeholders object for feeding
    num_time_steps -- number of graphs to train +1
    context_pairs -- list of (target, context) pairs obtained from random walk sampling.
    batch_size -- size of the minibatches (# nodes)
    """

    def __init__(self, graph, feat, adj, prev_hidden_embeds, placeholders, context_pairs=None, batch_size=100):

        self.graph = graph
        self.feat = feat
        self.adj = adj
        self.prev_hidden_embeds = prev_hidden_embeds
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.batch_num = 0
        self.degs = self.construct_degs()
        self.context_pairs = context_pairs
        self.max_positive = FLAGS.neg_sample_size
        self.train_nodes = self.graph.nodes()  # all nodes in the graph.
        print("# train nodes", len(self.train_nodes))

    def construct_degs(self):
        G = self.graph
        deg = np.zeros((len(G.nodes()),))
        for nodeid in G.nodes():
            neighbors = np.array(list(G.neighbors(nodeid)))
            deg[nodeid] = len(neighbors)
        return deg

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes):
        node_1 = []
        node_2 = []

        for n in batch_nodes:
            if len(self.context_pairs[n]) > self.max_positive:
                node_1.extend([n] * self.max_positive)
                node_2.extend(np.random.choice(self.context_pairs[n], self.max_positive, replace=False))
            else:
                node_1.extend([n] * len(self.context_pairs[n]))
                node_2.extend(self.context_pairs[n])

        assert len(node_1) == len(node_2)
        assert len(node_1) <= self.batch_size * self.max_positive
        feed_dict = dict()

        feed_dict.update({self.placeholders['node_1']: node_1})
        feed_dict.update({self.placeholders['node_2']: node_2})
        feed_dict.update({self.placeholders['feature']: self.feat})
        feed_dict.update({self.placeholders['adj']: self.adj})
        feed_dict.update({self.placeholders['prev_hidden_embeds'][t]: self.prev_hidden_embeds[t]
                          for t in range(0, len(self.prev_hidden_embeds))})

        feed_dict.update({self.placeholders['batch_nodes']: np.array(batch_nodes).astype(np.int32)})
        return feed_dict

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        return self.batch_feed_dict(batch_nodes)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        self.train_nodes = self.graph.nodes()
        self.batch_num = 0
