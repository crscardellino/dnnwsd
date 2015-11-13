# -*- coding: utf-8 -*-

import logging
import numpy as np

from keras import models, regularizers
from keras.layers import containers, core
from keras.utils import np_utils

from .base import BaseModel
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class MultiLayerPerceptron(BaseModel):
    _FIRST_LAYER = dict(
        hi=1600,
        mid=1000,
        lo=800,
        other=100
    )

    _SECOND_LAYER = dict(
        hi=800,
        mid=500,
        lo=400,
        other=50
    )

    def __init__(self, input_dim, output_dim, layers, fine_tune_epochs=10, pre_train_epochs=0, **kwargs):
        assert 1 <= layers <= 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._fine_tune_epochs = fine_tune_epochs
        self._pre_train_epochs = pre_train_epochs

        self._hidden_layers = self.__hidden_layers__(input_dim, layers)

        self._activation = kwargs.get('activation', 'tanh')
        self._optimizer = kwargs.get('optimizer', 'adam')
        self._batch_size = kwargs.get('batch_size', 64)
        self._weight_init = kwargs.get('weight_init', 'uniform')
        self._dropout_ratio = kwargs.get('dropout_ratio', 0.5)
        self._l1_regularizer = kwargs.get('l1_regularizer', 0.01)
        self._l2_regularizer = kwargs.get('l2_regularizer', 0.01)
        self._model = models.Sequential()

    def __hidden_layers__(self, input_dim, layers):
        if 3000 <= input_dim:
            layer_type = 'hi'
        elif 2000 <= input_dim <= 3000:
            layer_type = 'mid'
        elif 1500 <= input_dim <= 2000:
            layer_type = 'lo'
        else:
            layer_type = 'other'

        hidden_layers = list()
        hidden_layers.append(self._FIRST_LAYER[layer_type])
        if layers > 1:
            hidden_layers.append(self._SECOND_LAYER[layer_type])

        return hidden_layers

    def _fit(self, X, y):
        logger.info(u"Building the network architecture")

        self._model = models.Sequential()

        previous_layer_size = self.input_dim

        for layer_size in self._hidden_layers:
            self._model.add(
                core.Dense(
                    input_dim=previous_layer_size,
                    output_dim=layer_size,
                    init=self._weight_init,
                    activation=self._activation
                )
            )
            self._model.add(
                core.Dropout(self._dropout_ratio, input_shape=(layer_size,))
            )
            previous_layer_size = layer_size

        self._model.add(
            core.Dense(
                input_dim=previous_layer_size,
                output_dim=self.output_dim,
                activation='softmax',
                init=self._weight_init,
                W_regularizer=regularizers.WeightRegularizer(l1=self._l1_regularizer, l2=self._l2_regularizer),
                activity_regularizer=regularizers.ActivityRegularizer(l1=self._l1_regularizer, l2=self._l2_regularizer)
            )
        )

        logger.info(u"Compiling the network")

        self._model.compile(optimizer=self._optimizer, loss='categorical_crossentropy')

        logger.info(u"Fitting the data to the network")

        self._model.fit(X, y, batch_size=self._batch_size, nb_epoch=self._fine_tune_epochs, show_accuracy=True)

    def _pre_train(self, X):
        logger.info(u"Pre-training the network")

        encoders = []

        layers = self._hidden_layers[:]  # Copy the hidden layers list
        layers.insert(0, self.input_dim)

        for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]), start=1):
            logger.info(u"Training layer {}: Input {} -> Output {}".format(i, n_in, n_out))

            autoencoder = models.Sequential()

            encoder = containers.Sequential()
            encoder.add(
                core.Dropout(self._dropout_ratio, input_shape=(self.input_dim,))
            )
            encoder.add(
                core.Dense(
                    input_dim=n_in,
                    output_dim=n_out,
                    activation=self._activation,
                    init=self._weight_init
                )
            )

            decoder = containers.Sequential()
            decoder.add(
                core.Dense(
                    input_dim=n_out,
                    output_dim=n_in,
                    activation=self._activation,
                    init=self._weight_init
                )
            )

            autoencoder.add(
                core.AutoEncoder(
                    encoder=encoder,
                    decoder=decoder,
                    output_reconstruction=False
                )
            )

            autoencoder.compile(optimizer=self._optimizer, loss='mean_squared_error')
            autoencoder.fit(X, X, batch_size=self._batch_size, nb_epoch=self._pre_train_epochs)

            # Store trained weight and update training data
            encoders.append(autoencoder.layers[0].encoder)
            X = autoencoder.predict(X)
            pass

        return encoders

    def _fine_tuning(self, X, y, encoders):
        self._model = models.Sequential()

        logger.info(u"Fine tuning of the neural network")

        for encoder in encoders:
            self._model.add(encoder)

        self._model.add(
            core.Dense(
                input_dim=self._hidden_layers[-1],
                output_dim=self.output_dim,
                activation='softmax',
                init=self._weight_init,
                W_regularizer=regularizers.WeightRegularizer(l1=self._l1_regularizer, l2=self._l2_regularizer),
                activity_regularizer=regularizers.ActivityRegularizer(l1=self._l1_regularizer, l2=self._l2_regularizer)
            )
        )

        self._model.compile(optimizer=self._optimizer, loss='categorical_crossentropy')

        self._model.fit(X, y, batch_size=self._batch_size, nb_epoch=self._fine_tune_epochs, show_accuracy=True)

    def _fit_with_pre_train(self, X, y):
        encoders = self._pre_train(np.copy(X))

        self._fine_tuning(X, y, encoders)

    def fit(self, X, y):
        if hasattr(X, 'todense'):  # Deals with sparse matrices
            X = X.todense()

        y = np_utils.to_categorical(y, self.output_dim)

        if self._pre_train_epochs > 0:
            self._fit_with_pre_train(X, y)
        else:
            self._fit(X, y)

    def predict(self, X):
        if hasattr(X, 'todense'):
            X = X.todense()

        return self._model.predict_classes(X)

    def predict_proba(self, X):
        if hasattr(X, 'todense'):
            X = X.todense()

        return self._model.predict(X)
