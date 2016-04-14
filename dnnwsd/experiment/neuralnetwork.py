# -*- coding: utf-8 -*-

import logging
import math
import numpy as np
import os
import shutil
import tensorflow as tf

from sklearn.metrics import accuracy_score
from .base import NeuralNetworkExperiment
from ..utils.setup_logging import setup_logging
from ..utils.dataset import DataSets

setup_logging()
logger = logging.getLogger(__name__)


class MultilayerPerceptron(NeuralNetworkExperiment):
    """
    Multilayer Perceptron experiments which emulates a clean encoder
    of a Ladder Network. Useful to check if we can overfit the training data.
    """
    def __init__(self, dataset_path_or_instance, layers, epochs, starter_learning_rate, noise_std,
                 train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1):
        super(MultilayerPerceptron, self).__init__(dataset_path_or_instance, epochs, starter_learning_rate,
                                                   train_ratio, test_ratio, validation_ratio)
        self._noise_std = noise_std

        self._layers = layers
        self._layers.insert(0, self._input_size)
        self._layers.append(self._output_size)
        self._L = len(self._layers) - 1  # size of layers ignoring input layer

        # build network and return cost function
        self._cost = self.__build_network__()

        # define the y function as the classification function
        self._y = self.__build_classifier__()

        # loss
        self._loss = -tf.reduce_mean(tf.reduce_sum(self._outputs*tf.log(self._cost), 1))

        # y_true and y_pred used to get the metrics
        self._y_true = tf.argmax(self._outputs, 1)
        self._y_pred = tf.argmax(self._y, 1)

        # train_step for the weight parameters, optimized with Adam
        self._learning_rate = tf.Variable(self._starter_learning_rate, trainable=False)
        self._train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*self._bn_assigns)
        with tf.control_dependencies([self._train_step]):
            self._train_step = tf.group(bn_updates)

    def __build_network__(self):
        logger.info(u"Building network")
        # input of the network (will be use to place the examples for training and classification)
        self._inputs = tf.placeholder(tf.float32, shape=(None, self._input_size))

        # output of the network (will be use to place the labels of the examples for training and testing)
        self._outputs = tf.placeholder(tf.float32)

        # lambda functions to create the biases and weight (matrices) variables of the network
        bi = lambda inits, size, name: tf.Variable(inits * tf.ones([size]), name=name)
        # a bias has an initialization value (generally either one or zero), a size (lenght of the vector)
        # and a name

        wi = lambda shape, name: tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])
        # a weight has a shape of the matrix (inputs from previous layers outputs of the next layer)
        # and is initialized as a random normal divided by the lenght of the previous layer.

        shapes = zip(self._layers[:-1], self._layers[1:])
        # the shape of each of the linear layers, is needed to build the structure of the network

        # define the weights, randomly initialized at first, for the encoder and the decoder.
        # also define the biases for shift and scale the normalized values of a batch
        self._weights = dict(
                W=[wi(s, "W") for s in shapes],  # encoder weights
                V=[wi(s[::-1], "V") for s in shapes],  # decoder weights
                beta=[bi(0.0, self._layers[l+1], "beta") for l in range(self._L)],
                # batch normalization parameter to shift the normalized value
                gamma=[bi(1.0, self._layers[l+1], "gamma") for l in range(self._L)]
                # batch normalization parameter to scale the normalized value
        )

        # calculates the moving averages of mean and variance, needed for batch
        # normalization of the decoder step at each layer
        self._ewma = tf.train.ExponentialMovingAverage(decay=0.99)
        # stores the updates to be made to average mean and variance
        self._bn_assigns = []

        # average mean and variance of all layers
        self._running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in self._layers[1:]]
        self._running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in self._layers[1:]]

        h = self._inputs + tf.random_normal(tf.shape(self._inputs)) * self._noise_std
        h_clean = self._inputs

        for l in range(1, self._L+1):
            logger.info(u"Layer {}: {} -> {}".format(l, self._layers[l-1], self._layers[l]))

            # pre-activation
            z_pre = tf.matmul(h, self._weights['W'][l-1])
            z_pre_clean = tf.matmul(h_clean, self._weights['W'][l-1])

            # batch normalization + update the average mean and variance
            # using batch mean and variance of annotated examples
            z_clean = self._update_batch_normalization(z_pre_clean, l)
            mean, var = tf.nn.moments(z_pre, axes=[0])
            z = (z_pre - mean) / tf.sqrt(var + tf.constant(1e-10))
            z += tf.random_normal(tf.shape(z_pre)) * self._noise_std

            if l == self._L:
                # use softmax activation in output layer
                h = tf.nn.softmax(self._weights['gamma'][l-1] * (z + self._weights["beta"][l-1]))
                h_clean = tf.nn.softmax(self._weights['gamma'][l-1] * (z_clean + self._weights["beta"][l-1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self._weights["beta"][l-1])
                h_clean = tf.nn.relu(z_clean + self._weights["beta"][l-1])

        return h

    def __build_classifier__(self):
        h = self._inputs
        for l in range(1, self._L+1):
            # pre-activation
            z_pre = tf.matmul(h, self._weights['W'][l-1])

            # obtain average mean and variance and use it to normalize the batch
            mean = self._ewma.average(self._running_mean[l-1])
            var = self._ewma.average(self._running_var[l-1])
            z = (z_pre - mean) / tf.sqrt(var + tf.constant(1e-10))

            if l == self._L:
                # use softmax activation in output layer
                h = tf.nn.softmax(self._weights['gamma'][l-1] * (z + self._weights["beta"][l-1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self._weights["beta"][l-1])

        return h

    def _update_batch_normalization(self, batch, l):
        # batch normalize + update average mean and variance of layer l
        mean, var = tf.nn.moments(batch, axes=[0])
        assign_mean = self._running_mean[l-1].assign(mean)
        assign_var = self._running_var[l-1].assign(var)
        self._bn_assigns.append(self._ewma.apply([self._running_mean[l-1], self._running_var[l-1]]))
        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)

    def run(self, results_path):
        logger.info(u"Running session")

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            feed_dicts = {
                'train': {
                    self._inputs: self._train_ds.data,
                    self._outputs: self._train_ds.one_hot_labels
                },
                'test': {
                    self._inputs: self._test_ds.data,
                    self._outputs: self._test_ds.one_hot_labels
                },
                'validation': {
                    self._inputs: self._validation_ds.data,
                    self._outputs: self._validation_ds.one_hot_labels
                }
            }

            logger.info(u"Training start")

            for dataset in ['train', 'test', 'validation']:
                feed_dict = feed_dicts[dataset]

                y_true, y_pred = sess.run(
                        [self._y_true, self._y_pred], feed_dict=feed_dict
                )

                self._add_result(y_true, y_pred, dataset)

                logger.info(u"Initial {} accuracy: {:.2f}".format(dataset, self._results[dataset][-1]))

                if dataset == 'train':
                    train_error = sess.run(self._loss, feed_dict=feed_dict)
                    self._results['train_error'].append(train_error)
                    logger.info(u"Initial train error: {:.2f}".format(self._results['train_error'][-1]))

            for epoch in xrange(1, self._epochs + 1):
                data, target = self._train_ds.next_batch(self._batch_size)

                sess.run(self._train_step, feed_dict={
                    self._inputs: data,
                    self._outputs: target
                })

                for dataset in ['train', 'validation']:
                    feed_dict = feed_dicts[dataset]

                    y_true, y_pred = sess.run(
                            [self._y_true, self._y_pred], feed_dict=feed_dict
                    )

                    self._add_result(y_true, y_pred, dataset)

                    if dataset == 'train':
                        train_error = sess.run(self._loss, feed_dict=feed_dict)
                        self._results['train_error'].append(train_error)

                    if epoch > 1 and (epoch % 10) == 0:
                        logger.info(
                            u"Epoch {} - {} accuracy: {:.2f}".format(epoch, dataset, self._results[dataset][-1])
                        )

                        if dataset == 'train':
                            logger.info(
                                u"Epoch {} - train error: {:.2f}".format(epoch, self._results['train_error'][-1])
                            )

            for dataset in ['train', 'test', 'validation']:
                feed_dict = feed_dicts[dataset]

                y_true, y_pred = sess.run(
                        [self._y_true, self._y_pred], feed_dict=feed_dict
                )

                self._add_result(y_true, y_pred, dataset)

                logger.info(u"Final {} accuracy: {:.2f}".format(dataset, self._results[dataset][-1]))

                if dataset == 'train':
                    train_error = sess.run(self._loss, feed_dict=feed_dict)
                    self._results['train_error'].append(train_error)
                    logger.info(u"Final train error: {:.2f}".format(self._results['train_error'][-1]))

            self.save_results(results_path)

        del sess


class ConvolutionalNeuralNetwork(NeuralNetworkExperiment):
    def __init__(self, dataset_path_or_instance, epochs, starter_learning_rate,
                 train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1,
                 window_size=11, word_vector_size=300, filter_sizes=None, num_filters=64,
                 l2_reg_lambda=0.01, shift_data=False):
        super(ConvolutionalNeuralNetwork, self).__init__(dataset_path_or_instance, epochs, starter_learning_rate,
                                                         train_ratio, test_ratio, validation_ratio)

        self._window_size = window_size
        self._word_vector_size = word_vector_size
        self._whole_word_size = self._input_size / self._window_size
        self._shift_data = shift_data
        self._filter_sizes = filter_sizes if filter_sizes is not None else [2, 3, 4]
        self._num_filters = num_filters
        self._l2_reg_lambda = l2_reg_lambda

        self.__build_network__()

        # y_true and y_pred used to get the metrics
        self._y_true = tf.argmax(self._outputs, 1)
        self._y_pred = tf.argmax(self._scores, 1)

        # train_step for the weight parameters, optimized with Adam
        self._learning_rate = tf.Variable(self._starter_learning_rate, trainable=False)
        self._train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

    def __build_network__(self):
        # Placeholders for input, output and dropout
        self._inputs = tf.placeholder(tf.float32, [None, self._window_size, self._whole_word_size], name="inputs")
        self._expanded_inputs = tf.expand_dims(self._inputs, -1)
        self._outputs = tf.placeholder(tf.float32, [None, self._output_size], name="outputs")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        pooled_outputs = []
        for i, filter_size in enumerate(self._filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, self._whole_word_size, 1, self._num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self._num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self._expanded_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self._window_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self._num_filters * len(self._filter_sizes)
        self._h_pool = tf.concat(3, pooled_outputs)
        self._h_pool_flat = tf.reshape(self._h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self._h_drop = tf.nn.dropout(self._h_pool_flat, self._dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, self._output_size], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self._output_size]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self._scores = tf.nn.xw_plus_b(self._h_drop, W, b, name="scores")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self._scores, self._outputs)
            self._loss = tf.reduce_mean(losses) + self._l2_reg_lambda * l2_loss

    def _get_dataset(self, dataset):
        if self._shift_data:
            dataset = DataSets.shift_pos(dataset, self._word_vector_size, self._window_size)

        return np.reshape(dataset, [dataset.shape[0], self._window_size, self._whole_word_size])

    def run(self, results_path):
        logger.info(u"Running session")

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            feed_dicts = {
                'train': {
                    self._inputs: self._get_dataset(self._train_ds.data),
                    self._outputs: self._train_ds.one_hot_labels,
                    self._dropout_keep_prob: 0.5
                },
                'test': {
                    self._inputs: self._get_dataset(self._test_ds.data),
                    self._outputs: self._test_ds.one_hot_labels,
                    self._dropout_keep_prob: 0.5
                },
                'validation': {
                    self._inputs: self._get_dataset(self._validation_ds.data),
                    self._outputs: self._validation_ds.one_hot_labels,
                    self._dropout_keep_prob: 0.5
                }
            }

            logger.info(u"Training start")

            for dataset in ['train', 'test', 'validation']:
                feed_dict = feed_dicts[dataset]

                y_true, y_pred = sess.run(
                    [self._y_true, self._y_pred], feed_dict=feed_dict
                )

                self._add_result(y_true, y_pred, dataset)

                logger.info(u"Initial {} accuracy: {:.2f}".format(dataset, self._results[dataset][-1]))

                if dataset == 'train':
                    train_error = sess.run(self._loss, feed_dict=feed_dict)
                    self._results['train_error'].append(train_error)
                    logger.info(u"Initial train error: {:.2f}".format(self._results['train_error'][-1]))

            for epoch in xrange(1, self._epochs + 1):
                data, target = self._train_ds.next_batch(self._batch_size)

                sess.run(self._train_step, feed_dict={
                    self._inputs: self._get_dataset(data),
                    self._outputs: target,
                    self._dropout_keep_prob: 0.5
                })

                for dataset in ['train', 'validation']:
                    feed_dict = feed_dicts[dataset]

                    y_true, y_pred = sess.run(
                        [self._y_true, self._y_pred], feed_dict=feed_dict
                    )

                    self._add_result(y_true, y_pred, dataset)

                    if dataset == 'train':
                        train_error = sess.run(self._loss, feed_dict=feed_dict)
                        self._results['train_error'].append(train_error)

                    if epoch > 1 and (epoch % 10) == 0:
                        logger.info(
                            u"Epoch {} - {} accuracy: {:.2f}".format(epoch, dataset, self._results[dataset][-1])
                        )

                        if dataset == 'train':
                            logger.info(
                                u"Epoch {} - train error: {:.2f}".format(epoch, self._results['train_error'][-1])
                            )

            for dataset in ['train', 'test', 'validation']:
                feed_dict = feed_dicts[dataset]

                y_true, y_pred = sess.run(
                    [self._y_true, self._y_pred], feed_dict=feed_dict
                )

                self._add_result(y_true, y_pred, dataset)

                logger.info(u"Final {} accuracy: {:.2f}".format(dataset, self._results[dataset][-1]))

                if dataset == 'train':
                    train_error = sess.run(self._loss, feed_dict=feed_dict)
                    self._results['train_error'].append(train_error)
                    logger.info(u"Final train error: {:.2f}".format(self._results['train_error'][-1]))

            self.save_results(results_path)

        del sess
