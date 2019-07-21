from __future__ import print_function
import numpy as np
import networkx as nx
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from random_walk import Graph_RandomWalk

flags = tf.app.flags
FLAGS = flags.FLAGS


def to_one_hot(labels, N, multilabel=False):
    """In: list of (nodeId, label) tuples, #nodes N
       Out: N * |label| matrix"""
    ids, labels = zip(*labels)
    lb = MultiLabelBinarizer()
    if not multilabel:
        labels = [[x] for x in labels]
    lbs = lb.fit_transform(labels)
    encoded = np.zeros((N, lbs.shape[1]))
    for i in range(len(ids)):
        encoded[ids[i]] = lbs[i]
    return encoded


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


"""Random walk-based pair generation."""

def run_random_walks_n2v(graph, nodes, num_walks=10, walk_len=40):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using the sampling strategy of node2vec (deepwalk)"""
    walk_len = FLAGS.walk_len
    nx_G = nx.Graph()
    adj = nx.adjacency_matrix(graph)
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])

    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 10
    pairs = defaultdict(lambda: [])
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs
