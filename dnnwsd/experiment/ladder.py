# -*- coding: utf-8 -*-

import logging
import math
import numpy as np
import os
import tensorflow as tf
import tqdm

from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ..utils.dataset import DataSets
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class LadderNetworksExperiment(object):
    def __init__(self, dataset_path, layers, denoising_cost, checkpoint_path,
                 epochs=50, noise_std=0.3, starter_learning_rate=0.02,
                 train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1):

        self._dataset = DataSets(dataset_path, train_ratio, test_ratio, validation_ratio)

        logger.info(u"Dataset for lemma {} loaded.".format(self._dataset.lemma))

        self._checkpoint_path = checkpoint_path

        self._input_size = self._dataset.train_ds.annotated_ds.vector_length
        self._output_size = self._dataset.train_ds.annotated_ds.labels_count

        self._layers = layers
        self._layers.insert(0, self._input_size)
        self._layers.append(self._output_size)
        self._L = len(self._layers) - 1  # size of layers ignoring input layer

        self._num_examples = self._dataset.train_ds.data_count
        self._num_epochs = epochs
        self._batch_size = self._dataset.train_ds.annotated_ds.data_count
        self._num_iter = (self._num_examples / self._batch_size) * self._num_epochs

        self._noise_std = noise_std  # scaling factor for noise used in corrupted encoder
        self._denoising_cost = denoising_cost  # hyperparameters that denote the importance of each layer

        # functions to join and split annotated and unannotated corpus
        self._join = lambda a, u: tf.concat(0, [a, u])
        self._annotated = lambda i: tf.slice(i, [0, 0], [self._batch_size, -1]) if i is not None else i
        self._unannotated = lambda i: tf.slice(i, [self._batch_size, 0], [-1, -1]) if i is not None else i
        self._split_lu = lambda i: (self._annotated(i), self._unannotated(i))

        self._build_network()

        logger.info(u"Building corrupted encoder")
        self._y_c, self._corrupted_encoder = self._encoder(self._inputs, self._noise_std)

        logger.info(u"Building clean encoder")
        _, self._clean_encoder = self._encoder(self._inputs, 0.0)
        # the y function is ignored as is only helpful for evaluation and classification

        # define the y function as the classification function
        self._y = self._mlp(self._inputs)

        logger.info(u"Building decoder and unannotated cost function")

        # calculate total unsupervised cost by adding the denoising cost of all layers
        self._ucost = tf.add_n(self._decoder())

        self._y_N = self._annotated(self._y_c)
        self._scost = -tf.reduce_mean(tf.reduce_sum(self._outputs*tf.log(self._y_N), 1))

        # total cost (loss function)
        self._loss = self._scost + self._ucost

        # prediction error of supervised dataset
        self._supervised_error = -tf.reduce_mean(tf.reduce_sum(self._outputs*tf.log(self._y), 1))

        # y_true and y_pred used to get the metrics
        self._y_true = tf.argmax(self._outputs, 1)
        self._y_pred = tf.argmax(self._y, 1)

        # dictionary of results
        if self._dataset.validation_ds:
            full_target = np.hstack((
                self._dataset.train_ds.annotated_ds.target,
                self._dataset.test_ds.target,
                self._dataset.validation_ds.target
            ))
        else:
            full_target = np.hstack((
                self._dataset.train_ds.annotated_ds.target,
                self._dataset.test_ds.target,
            ))

        self._target_counts = [c[0] for c in Counter(full_target).most_common()]

        self._results = dict(
            initial_accuracy=0,
            final_accuray=0,
            validation_accuracy=[],
            initial_mcp=0,
            final_mcp=0,
            validation_mcp=[],
            initial_lcr=0,
            final_lcr=0,
            validation_lcr=[],
            supervised_error=[]
        )

        # evaluation_sentences
        self._evaluation_sentences = []

        # train_step for the weight parameters, optimized with Adam
        self._learning_rate = tf.Variable(starter_learning_rate, trainable=False)
        self._train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*self._bn_assigns)
        with tf.control_dependencies([self._train_step]):
            self._train_step = tf.group(bn_updates)

        self._run()  # Run the training

    @property
    def results(self):
        return self._results

    @property
    def evaluation_sentences(self):
        return self._evaluation_sentences

    @property
    def dataset(self):
        return self._dataset

    def _add_result(self, y_true, y_pred, supervised_error, phase):
        assert phase in {'initial', 'final', 'validation'}

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(len(self._dataset.labels))
        )
        mcp = precision[self._target_counts[0]]
        lcr = recall[self._target_counts[1:]].mean()

        if phase == 'initial' or phase == 'final':
            self._results['{}_accuracy'.format(phase)] = accuracy
            self._results['{}_mcp'.format(phase)] = mcp
            self._results['{}_lcr'.format(phase)] = lcr
        else:
            self._results['validation_accuracy'].append(accuracy)
            self._results['validation_mcp'].append(mcp)
            self._results['validation_lcr'].append(lcr)

        self._results['supervised_error'].append(supervised_error)

    def _build_network(self):
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

    def _update_batch_normalization(self, batch, l):
        # batch normalize + update average mean and variance of layer l
        mean, var = tf.nn.moments(batch, axes=[0])
        assign_mean = self._running_mean[l-1].assign(mean)
        assign_var = self._running_var[l-1].assign(var)
        self._bn_assigns.append(self._ewma.apply([self._running_mean[l-1], self._running_var[l-1]]))
        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)

    def _decoder(self):
        z_est = {}
        # stores the denoising cost of all layers
        d_cost = []

        for l in range(self._L, -1, -1):
            logger.info(u"Layer {}: {} -> {}, denoising cost: {}".format(
                l,
                self._layers[l+1] if l+1 < len(self._layers) else None,
                self._layers[l],
                self._denoising_cost[l]
            ))

            z, z_c = self._clean_encoder['unannotated']['z'][l], self._corrupted_encoder['unannotated']['z'][l]

            m = self._clean_encoder['unannotated']['m'].get(l, 0)
            v = self._clean_encoder['unannotated']['v'].get(l, 1-1e-10)

            if l == self._L:
                u = self._unannotated(self._y_c)
            else:
                u = tf.matmul(z_est[l+1], self._weights['V'][l])

            u = self._batch_normalization(u)
            z_est[l] = self._combinator_g(z_c, u, self._layers[l])
            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            d_cost.append(
                    (tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / self._layers[l]) *
                    self._denoising_cost[l]
            )

        return d_cost

    def _encoder(self, inputs, noise_std):
        """
        encoder factory for training.
        """
        # add noise to input
        h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std

        # dictionary to store the pre-activation, activation, mean and variance for each layer
        layer_data = dict()

        # the data for labeled and unlabeled examples are stored separately
        layer_data['annotated'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        layer_data['unannotated'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

        # get the data for the input layer, divided in annotated and unannotated
        layer_data['annotated']['z'][0], layer_data['unannotated']['z'][0] = self._split_lu(h)
        for l in range(1, self._L+1):
            logger.info(u"Layer {}: {} -> {}".format(l, self._layers[l-1], self._layers[l]))
            layer_data['annotated']['h'][l - 1], layer_data['unannotated']['h'][l - 1] = self._split_lu(h)

            # pre-activation
            z_pre = tf.matmul(h, self._weights['W'][l-1])
            # split annotated and unannotated examples
            z_pre_l, z_pre_u = self._split_lu(z_pre)

            # batch normalization for annotated and unannotated examples is performed separately
            m, v = tf.nn.moments(z_pre_u, axes=[0])
            if noise_std > 0:
                # Corrupted encoder
                # batch normalization + noise
                z = self._join(self._batch_normalization(z_pre_l), self._batch_normalization(z_pre_u, m, v))
                z += tf.random_normal(tf.shape(z_pre)) * noise_std
            else:
                # Clean encoder
                # batch normalization + update the average mean and variance
                # using batch mean and variance of annotated examples
                z = self._join(self._update_batch_normalization(z_pre_l, l), self._batch_normalization(z_pre_u, m, v))

            if l == self._L:
                # use softmax activation in output layer
                h = tf.nn.softmax(self._weights['gamma'][l-1] * (z + self._weights["beta"][l-1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self._weights["beta"][l-1])

            layer_data['annotated']['z'][l], layer_data['unannotated']['z'][l] = self._split_lu(z)

            # save mean and variance of unannotated examples for decoding
            layer_data['unannotated']['m'][l], layer_data['unannotated']['v'][l] = m, v

        # get the h values for unannotated and annotated for the last layer
        layer_data['annotated']['h'][l], layer_data['unannotated']['h'][l] = self._split_lu(h)

        return h, layer_data

    def _mlp(self, inputs):
        h = inputs
        for l in range(1, self._L+1):
            logger.info(u"Layer {}: {} -> {}".format(l, self._layers[l-1], self._layers[l]))

            # pre-activation
            z_pre = tf.matmul(h, self._weights['W'][l-1])

            # obtain average mean and variance and use it to normalize the batch
            mean = self._ewma.average(self._running_mean[l-1])
            var = self._ewma.average(self._running_var[l-1])
            z = self._batch_normalization(z_pre, mean, var)

            if l == self._L:
                # use softmax activation in output layer
                h = tf.nn.softmax(self._weights['gamma'][l-1] * (z + self._weights["beta"][l-1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self._weights["beta"][l-1])
        return h

    @staticmethod
    def _batch_normalization(batch, mean=None, var=None):
        if mean == None or var == None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    @staticmethod
    def _combinator_g(z_c, u, size):
        """
        combinator function for the lateral z_c and the vertical u value in each
        layer of the decoder, proposed by the original paper
        """
        wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est

    def _run(self):
        logger.info(u"Running session")

        with tf.Session() as sess:
            saver = tf.train.Saver()
            i_iter = 0

            # get latest checkpoint (if any)
            ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)

            if ckpt and ckpt.model_checkpoint_path:
                logger.info(u"Restoring training session from checkpoint")
                # if checkpoint exists, restore the parameters and set epoch_n and i_iter
                saver.restore(sess, ckpt.model_checkpoint_path)
                epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])

                i_iter = (epoch_n+1) * (self._num_examples/self._batch_size)
                logger.info(u"Restored Epoch {}".format(epoch_n))
            else:
                # no checkpoint exists. create checkpoints directory if it does not exist.
                if not os.path.exists(self._checkpoint_path):
                    os.makedirs(self._checkpoint_path)
                init = tf.initialize_all_variables()
                sess.run(init)

            test_dict = {
                self._inputs: self._dataset.test_ds.data,
                self._outputs: self._dataset.test_ds.one_hot_labels
            }

            if self._dataset.validation_ds:
                validation_dict = {
                    self._inputs: self._dataset.validation_ds.data,
                    self._outputs: self._dataset.validation_ds.one_hot_labels
                }

            logger.info(u"Training start")

            y_true, y_pred, s_error = sess.run(
                [self._y_true, self._y_pred, self._supervised_error], feed_dict=test_dict
            )
            self._add_result(y_true, y_pred, s_error, 'initial')
            logger.info(u"Initial test accuracy: {:.2f}".format(self._results['initial_accuracy']))
            logger.info(u"Initial test mcp: {:.2f}".format(self._results['initial_mcp']))
            logger.info(u"Initial test lcr: {:.2f}".format(self._results['initial_lcr']))
            logger.info(u"Initial supervised error: {:.2f}".format(self._results['supervised_error'][0]))

            for i in tqdm.tqdm(range(i_iter, self._num_iter)):
                data, target = self._dataset.train_ds.next_batch(self._batch_size)

                sess.run(self._train_step, feed_dict={
                    self._inputs: data,
                    self._outputs: target
                })

                if (i > 1) and ((i+1) % (self._num_iter/self._num_epochs) == 0):
                    epoch_n = i/(self._num_examples/self._batch_size)
                    saver.save(sess, '{}/model.ckpt'.format(self._checkpoint_path), epoch_n)

                    if self._dataset.validation_ds:
                        y_true, y_pred, s_error = sess.run(
                            [self._y_true, self._y_pred, self._supervised_error], feed_dict=validation_dict
                        )
                        self._add_result(y_true, y_pred, s_error, 'validation')
                        logger.info(u"Epoch {} - Validation accuracy: {:.2f}"
                                    .format(epoch_n, self._results['validation_accuracy'][-1]))
                        logger.info(u"Epoch {} - Validation mcp: {:.2f}"
                                    .format(epoch_n, self._results['validation_mcp'][-1]))
                        logger.info(u"Epoch {} - Validation lcr: {:.2f}"
                                    .format(epoch_n, self._results['validation_lcr'][-1]))
                        logger.info(u"Epoch {} - Validation supervised error: {:.2f}"
                                    .format(epoch_n, self._results['supervised_error'][-1]))

                    logger.info(u"Selecting unannotated data for manual evaluation")
                    # selecting 10 random unannotated instances for classification and manual evaluation
                    perm = np.arange(self._dataset.train_ds.unannotated_ds.data_count)
                    np.random.shuffle(perm)
                    perm = perm[:10]
                    eval_data = self._dataset.train_ds.unannotated_ds.data[perm]
                    eval_sent = self._dataset.train_ds.unannotated_ds.target[perm]
                    y_pred = sess.run(self._y_pred, feed_dict={self._inputs: eval_data})
                    self._evaluation_sentences.append(
                        zip(eval_sent, y_pred)
                    )

            y_true, y_pred, s_error = sess.run(
                [self._y_true, self._y_pred, self._supervised_error], feed_dict=test_dict
            )
            self._add_result(y_true, y_pred, s_error, 'final')
            logger.info(u"Final test accuracy: {:.2f}".format(self._results['final_accuracy']))
            logger.info(u"Final test mcp: {:.2f}".format(self._results['final_mcp']))
            logger.info(u"Final test lcr: {:.2f}".format(self._results['final_lcr']))
            logger.info(u"Final supervised error: {:.2f}".format(self._results['supervised_error'][-1]))

