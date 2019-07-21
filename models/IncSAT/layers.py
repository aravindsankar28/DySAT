from tensorflow.python.ops import math_ops
from inits import *

conv1d = tf.layers.conv1d

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class TemporalAttentionLayer(Layer):
    """ The number of time steps is the total number of training snapshots +1. """

    def __init__(self, input_dim, n_heads, max_time_steps, attn_drop, residual=False,
                 bias=True, use_position_embedding=True, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

        self.bias = bias
        self.n_heads = n_heads
        self.max_time_steps = max_time_steps
        self.attn_drop = attn_drop
        self.attn_wts_means = []
        self.attn_wts_vars = []
        self.residual = residual
        xavier_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name + '_vars'):
            if use_position_embedding:
                self.vars['position_embeddings'] = tf.get_variable('position_embeddings',
                                                                   dtype=tf.float32,
                                                                   shape=[self.max_time_steps, input_dim],
                                                                   initializer=xavier_init)  # [T, F]

            self.vars['Q_embedding_weights'] = tf.get_variable('Q_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]
            self.vars['K_embedding_weights'] = tf.get_variable('K_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]
            self.vars['V_embedding_weights'] = tf.get_variable('V_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]

    def __call__(self, inputs):
        """ Computes multi-head temporal self-attention with positional embeddings."""

        # 1: Add position embeddings to input.
        position_inputs = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1])
        temporal_inputs = inputs + tf.nn.embedding_lookup(self.vars['position_embeddings'],
                                                          position_inputs)  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = tf.tensordot(temporal_inputs, self.vars['Q_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        k = tf.tensordot(temporal_inputs, self.vars['K_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        v = tf.tensordot(temporal_inputs, self.vars['V_embedding_weights'], axes=[[2], [0]])  # [N, T, F]

        # 3: Split, concat and scale.
        q_ = tf.concat(tf.split(q, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        k_ = tf.concat(tf.split(k, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        v_ = tf.concat(tf.split(v, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]

        outputs = tf.matmul(q_, tf.transpose(k_, [0, 2, 1]))  # [hN, T, T]
        outputs = outputs / (k_.get_shape().as_list()[-1] ** 0.5)

        diag_val = tf.ones_like(outputs[0, :, :])  # [T, T]
        tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # [hN, T, T]
        padding = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), padding, outputs)  # [h*N, T, T]
        outputs = tf.nn.softmax(outputs)  # Masked attention.
        self.attn_wts_all = outputs

        # 5: Dropout on attention weights.
        outputs = tf.layers.dropout(outputs, rate=self.attn_drop)
        outputs = tf.matmul(outputs, v_)  # [hN, T, C/h]

        split_outputs = tf.split(outputs, self.n_heads, axis=0)
        outputs = tf.concat(split_outputs, axis=-1)

        # Optional: Feedforward and residual
        if FLAGS.position_ffn:
            outputs = self.feedforward(outputs)

        if self.residual:
            outputs += temporal_inputs

        return outputs

    def feedforward(self, inputs, reuse=None):
        """Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        """
        with tf.variable_scope(self.name + '_vars', reuse=reuse):
            params = {"inputs": inputs, "filters": inputs.get_shape()[-1], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs += inputs
        return outputs


# Has multiple graph attention heads.
class StructuralAttentionLayer(Layer):
    '''GAT layer.
    Args: Requires input and output dimensions and # attention heads (n_heads) in the layer.
    attn_drop -> For co-efficients, ffd_drop -> For embeddings, however we use the same dropout rate for both.
    sparse_inputs is generally True for the first GAT layer.'''

    def __init__(self, input_dim, output_dim, n_heads, attn_drop, ffd_drop, act=tf.nn.elu, residual=False,
                 bias=True, sparse_inputs=False, **kwargs):
        super(StructuralAttentionLayer, self).__init__(**kwargs)
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.act = act
        self.bias = bias
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual
        self.sparse_inputs = sparse_inputs

        if self.logging:
            self._log_vars()
        self.n_calls = 0

    def _call(self, inputs):
        self.n_calls += 1
        x = inputs[0]
        adj = inputs[1]
        attentions = []
        reuse_scope = None
        for j in range(self.n_heads):
            if self.n_calls > 1:
                reuse_scope = True

            attentions.append(self.sp_attn_head(x, adj_mat=adj, in_sz=self.input_dim,
                                                out_sz=self.output_dim // self.n_heads, activation=self.act,
                                                in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual,
                                                layer_str="l_{}_h_{}".format(self.name, j),
                                                sparse_inputs=self.sparse_inputs,
                                                reuse_scope=reuse_scope))

        h = tf.concat(attentions, axis=-1)
        return h

    @staticmethod
    def leaky_relu(features, alpha=0.2):
        return math_ops.maximum(alpha * features, features)

    def sp_attn_head(self, seq, in_sz, out_sz, adj_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
                     layer_str="", sparse_inputs=False, reuse_scope=None):
        """ Sparse Attention Head for the GAT layer. Note: the variable scope is necessary to avoid
        variable duplication across snapshots"""

        with tf.variable_scope('struct_attn', reuse=reuse_scope):
            if sparse_inputs:
                weight_var = tf.get_variable("layer_" + str(layer_str) + "_weight_transform", shape=[in_sz, out_sz],
                                             dtype=tf.float32)
                seq_fts = tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, weight_var), axis=0)  # [N, F]
            else:
                seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False,
                                           name='layer_' + str(layer_str) + '_weight_transform', reuse=reuse_scope)

            # Additive self-attention.
            f_1 = tf.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a1', reuse=reuse_scope)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a2', reuse=reuse_scope)
            f_1 = tf.reshape(f_1, [-1, 1])  # [N, 1]
            f_2 = tf.reshape(f_2, [-1, 1])  # [N, 1]

            logits = tf.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2))  # adj_mat is [N, N] (sparse)

            leaky_relu = tf.SparseTensor(indices=logits.indices,
                                         values=self.leaky_relu(logits.values),
                                         dense_shape=logits.dense_shape)
            coefficients = tf.sparse_softmax(leaky_relu)  # [N, N] (sparse)

            if coef_drop != 0.0:
                coefficients = tf.SparseTensor(indices=coefficients.indices,
                                               values=tf.nn.dropout(coefficients.values, 1.0 - coef_drop),
                                               dense_shape=coefficients.dense_shape)  # [N, N] (sparse)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)  # [N, D]

            seq_fts = tf.squeeze(seq_fts)
            values = tf.sparse_tensor_dense_matmul(coefficients, seq_fts)
            values = tf.reshape(values, [-1, out_sz])
            values = tf.expand_dims(values, axis=0)
            ret = values  # [1, N, F]

            if residual:
                residual_wt = tf.get_variable("layer_" + str(layer_str) + "_residual_weight", shape=[in_sz, out_sz],
                                              dtype=tf.float32)
                if sparse_inputs:
                    ret = ret + tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, residual_wt),
                                               axis=0)  # [N, F] * [F, D] = [N, D].
                else:
                    ret = ret + tf.layers.conv1d(seq, out_sz, 1, use_bias=False,
                                                 name='layer_' + str(layer_str) + '_residual_weight', reuse=reuse_scope)
            return activation(ret)
