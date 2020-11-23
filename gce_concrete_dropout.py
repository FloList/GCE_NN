"""
Concrete dropout class for the GCE NN (adapted from Y. Gal's github).
https://github.com/yaringal/ConcreteDropout
"""

import tensorflow as tf
import numpy as np


class ConcreteDropout:
    def __init__(self, graph, input, weights, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_conv_layer=False):
        with graph.as_default():
            # Concrete dropout
            with tf.compat.v1.name_scope('concrete_dropout'):
                self.weights = weights
                self.input = input  # for conv: N x (Fin * K)
                self.input_drop = []
                self.weight_regularizer = weight_regularizer
                self.dropout_regularizer = dropout_regularizer
                self.p_logit = None
                self.p = None
                self.init_min = (np.log(init_min) - np.log(1. - init_min))
                self.init_max = (np.log(init_max) - np.log(1. - init_max))
                self.is_conv_layer = is_conv_layer
                self.build()
                self.concrete_dropout()

    def get_input_drop_and_weights(self):
        return self.input_drop, self.weights

    def build(self):

        # initialise p
        self.p_logit = tf.compat.v1.get_variable('p_logit', shape=[1],
                                                 initializer=tf.compat.v1.random_uniform_initializer(self.init_min, self.init_max),
                                                 dtype=tf.float32, trainable=True)

        self.p = tf.nn.sigmoid(self.p_logit[0])
        tf.compat.v1.add_to_collection("LAYER_P", self.p)
        tf.compat.v1.summary.scalar("dropout_p", self.p)

        # initialise regulariser / prior KL term
        if self.is_conv_layer:
            input_dim = self.input.shape[2]  # input has shape: M x N x Fin*K
        else:
            input_dim = self.input.shape[1]  # input has shape: N x Fin

        weight = self.weights
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
            weight)) / (1. - self.p)
        dropout_regularizer = self.p * tf.math.log(self.p)
        dropout_regularizer += (1. - self.p) * tf.math.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        # Add the regularisation loss to collection.
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    def concrete_dropout(self):

        eps = 1e-7
        temp = 2. / 3. if self.is_conv_layer else 0.1

        if self.is_conv_layer:
            unif_noise = tf.random.uniform(shape=[1] + self.input.shape[1:].as_list())  # 1 x N x Fin*K (same for all pix m)
        else:
            unif_noise = tf.random.uniform(shape=tf.shape(self.input))  # N x Fin

        drop_prob = (
            tf.math.log(self.p + eps)
            - tf.math.log(1. - self.p + eps)
            + tf.math.log(unif_noise + eps)
            - tf.math.log(1. - unif_noise + eps)
        )
        drop_prob = tf.nn.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        self.input_drop = self.input * random_tensor / retain_prob

