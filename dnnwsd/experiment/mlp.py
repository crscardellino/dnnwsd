# -*- coding: utf-8 -*-

import logging
import math
import numpy as np
import os
import shutil
import tensorflow as tf

from sklearn.metrics import accuracy_score
from ..utils.dataset import DataSets
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class MultilayerPerceptron(object):
    """
    Multilayer Perceptron experiments which emulates a clean encoder
    of a Ladder Network. Useful to check if we can overfit the training data.
    """
    def __init__(self, dataset_path, layers, epochs, starter_learning_rate, noise_std,
                 train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1):
        if type(dataset_path) == str:
            dataset = DataSets(dataset_path, train_ratio, test_ratio, validation_ratio)
        elif type(dataset_path) == DataSets:
            dataset = dataset_path
        else:
            raise Exception("The provided dataset is not valid")

        self._lemma = dataset.lemma
        self._train_ds = dataset.train_ds.annotated_ds
        self._test_ds = dataset.test_ds
        self._validation_ds = dataset.validation_ds

        self._noise_std = noise_std

        logger.info(u"Dataset for lemma {} loaded.".format(self._lemma))

        self._input_size = self._train_ds.vector_length
        self._output_size = self._train_ds.labels_count

        self._layers = layers
        self._layers.insert(0, self._input_size)
        self._layers.append(self._output_size)
        self._L = len(self._layers) - 1  # size of layers ignoring input layer

        self._num_examples = self._train_ds.data_count
        self._batch_size = self._train_ds.data_count
        self._epochs = epochs

        # build network and return cost function
        self._cost = self._build_netword()

        # define the y function as the classification function
        self._y = self._mlp()

        # loss
        self._loss = -tf.reduce_mean(tf.reduce_sum(self._outputs*tf.log(self._cost), 1))

        # y_true and y_pred used to get the metrics
        self._y_true = tf.argmax(self._outputs, 1)
        self._y_pred = tf.argmax(self._y, 1)

        self._results = dict(
            train=[],
            train_error=[],
            test=[],
            validation=[]
        )

        # train_step for the weight parameters, optimized with Adam
        self._learning_rate = tf.Variable(starter_learning_rate, trainable=False)
        self._train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*self._bn_assigns)
        with tf.control_dependencies([self._train_step]):
            self._train_step = tf.group(bn_updates)

    @property
    def results(self):
        return self._results

    def _build_netword(self):
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

    def _mlp(self):
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

        with tf.Session() as sess:
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

                self._results[dataset].append(accuracy_score(y_true, y_pred))

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

                    self._results[dataset].append(accuracy_score(y_true, y_pred))

                    if epoch > 1 and (epoch % 10) == 0:
                        logger.info(
                            u"Epoch {} - {} accuracy: {:.2f}".format(epoch, dataset, self._results[dataset][-1])
                        )

                        if dataset == 'train':
                            train_error = sess.run(self._loss, feed_dict=feed_dict)
                            self._results['train_error'].append(train_error)
                            logger.info(
                                u"Epoch {} - train error: {:.2f}".format(epoch, self._results['train_error'][-1])
                            )

            for dataset in ['train', 'test', 'validation']:
                feed_dict = feed_dicts[dataset]

                y_true, y_pred = sess.run(
                        [self._y_true, self._y_pred], feed_dict=feed_dict
                )

                self._results[dataset].append(accuracy_score(y_true, y_pred))

                logger.info(u"Final {} accuracy: {:.2f}".format(dataset, self._results[dataset][-1]))

                if dataset == 'train':
                    train_error = sess.run(self._loss, feed_dict=feed_dict)
                    self._results['train_error'].append(train_error)
                    logger.info(u"Final train error: {:.2f}".format(self._results['train_error'][-1]))

            if os.path.exists(results_path):
                shutil.rmtree(results_path)

            os.makedirs(results_path)

            for dataset in ['train', 'test', 'validation']:
                np.savetxt(
                    os.path.join(results_path, dataset), np.array(self._results[dataset], dtype=np.float32), fmt="%.2f"
                )

            np.savetxt(
                os.path.join(results_path, 'train_error'), np.array(self._results['train_error'], dtype=np.float32),
                fmt="%.2f"
            )
