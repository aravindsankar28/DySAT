from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        # Build metrics
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class IncSAT(Model):
    def __init__(self, placeholders, num_features, num_features_nonzero, degrees, **kwargs):
        super(IncSAT, self).__init__(**kwargs)
        self.temporal_attention_layers = []
        self.structural_attention_layers = []
        self.placeholders = placeholders
        self.num_time_steps = len(placeholders['prev_hidden_embeds'])+1
        self.num_time_steps_train = self.num_time_steps
        self.num_features = num_features
        self.num_features_nonzero = num_features_nonzero
        self.degrees = degrees
        self.num_features = num_features
        self.structural_head_config = map(int, FLAGS.structural_head_config.split(","))
        self.structural_layer_config = map(int, FLAGS.structural_layer_config.split(","))
        self.temporal_head_config = map(int, FLAGS.temporal_head_config.split(","))
        self.temporal_layer_config = map(int, FLAGS.temporal_layer_config.split(","))
        self._build()

    def _build(self):
        proximity_labels = tf.expand_dims(tf.cast(self.placeholders['node_2'], tf.int64), 1)
        self.proximity_neg_samples = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=proximity_labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist())[0]

        # Build neural network.
        self.final_output_embeddings = self.build_net(self.structural_head_config, self.structural_layer_config,
                                                      self.temporal_head_config, self.temporal_layer_config,
                                                      self.placeholders['spatial_drop'],
                                                      self.placeholders['temporal_drop'], self.placeholders['adj'])
        self._loss()
        self.init_optimizer()

    def build_net(self, attn_head_config, attn_layer_config, temporal_head_config, temporal_layer_config,
                  spatial_drop, temporal_drop, adj):
        input_dim = self.num_features
        sparse_inputs = True  # Sparse features as input to the first layer.

        # 1: Structural Attention Layers
        for i in range(0, len(attn_layer_config)):
            if i > 0:
                input_dim = attn_layer_config[i-1]
                sparse_inputs = False
            self.structural_attention_layers.append(StructuralAttentionLayer(input_dim=input_dim,
                                                                             output_dim=attn_layer_config[i],
                                                                             n_heads=attn_head_config[i],
                                                                             attn_drop=spatial_drop,
                                                                             ffd_drop=spatial_drop, act=tf.nn.elu,
                                                                             sparse_inputs = sparse_inputs,
                                                                             residual=False))
        # 2: Temporal Attention Layers
        input_dim = attn_layer_config[-1]
        for i in range(0, len(temporal_layer_config)):
            if i > 0:
                input_dim = temporal_layer_config[i-1]
            temporal_layer = TemporalAttentionLayer(input_dim=input_dim, n_heads=temporal_head_config[i],
                                                    attn_drop=temporal_drop, max_time_steps=100, residual=False)
            self.temporal_attention_layers.append(temporal_layer)

        # 3: Structural Attention forward
        input_list = self.placeholders['feature']  # List of T feature matrices. [N x F]
        for idx, layer in enumerate(self.structural_attention_layers):
            out = layer([input_list, adj])
            input_list = out

        # 4: Pack embeddings across snapshots.
        hidden_layer_embeds = []
        hidden_layer_embeds.extend(self.placeholders['prev_hidden_embeds'])
        hidden_layer_embeds.append(out)

        for t in range(0, self.num_time_steps):
            zero_padding = tf.zeros([1, tf.shape(hidden_layer_embeds[-1])[1] - tf.shape(hidden_layer_embeds[t])[1] ,
                                     attn_layer_config[-1]])
            hidden_layer_embeds[t] = tf.concat([hidden_layer_embeds[t], zero_padding], axis=1)

        self.hidden_embeds = hidden_layer_embeds  # Saved as class variable for later use.
        if self.num_time_steps > 1:
            structural_outputs = tf.transpose(tf.concat(hidden_layer_embeds, axis=0), [1, 0, 2])  # [N, T, F]
        else:
            structural_outputs = tf.reshape(out, [-1, 1, attn_layer_config[-1]])
        structural_outputs = tf.reshape(structural_outputs, [-1, self.num_time_steps, attn_layer_config[-1]])  # [
        # N, T, F]

        # 5: Temporal Attention forward
        temporal_inputs = structural_outputs

        for temporal_layer in self.temporal_attention_layers:
            outputs = temporal_layer(temporal_inputs)  # [N, T, F]
            temporal_inputs = outputs
        return outputs

    def _loss(self):
        """ Random-walk based graph context prediction loss over latest snapshot"""
        self.graph_loss = tf.constant(0.0)
        time_idx = self.num_time_steps_train - 1
        output_embeds_t = tf.nn.embedding_lookup(tf.transpose(self.final_output_embeddings, [1, 0, 2]), time_idx)
        inputs1 = tf.nn.embedding_lookup(output_embeds_t, self.placeholders['node_1'])
        inputs2 = tf.nn.embedding_lookup(output_embeds_t, self.placeholders['node_2'])
        pos_score = tf.reduce_sum(inputs1 * inputs2, axis=1)
        neg_samples = tf.nn.embedding_lookup(output_embeds_t, self.proximity_neg_samples)
        neg_score = (-1.0)*tf.matmul(inputs1, tf.transpose(neg_samples))
        pos_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_score), logits=pos_score)
        neg_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(neg_score), logits=neg_score)
        self.graph_loss += tf.reduce_mean(pos_ent) + tf.reduce_mean(neg_ent)

        self.reg_loss = tf.constant(0.0)
        self.reg_loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                   if "struct_attn" in v.name and "bias" not in v.name]) * FLAGS.weight_decay
        self.loss = self.graph_loss + self.reg_loss

    def init_optimizer(self):
        trainable_params = tf.trainable_variables()
        actual_loss = self.loss
        gradients = tf.gradients(actual_loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # Set the model optimization op.
        self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    def _accuracy(self):
        pass
