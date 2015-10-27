# -*- coding: utf-8 -*-

import logging

from keras import models, regularizers
from keras.layers import containers, core
from keras.utils import np_utils

from .base import BaseModel
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class DenoisingAutoencoder(BaseModel):
    def __init__(self, input_size, layer, activation, classes_amount, pre_train_epochs=5,
                 fine_tune_epochs=10, batch_size=64):
        self._input_size = input_size
        self._layer = layer
        self._activation = activation
        self._classes_amount = classes_amount
        self._pre_train_epochs = pre_train_epochs
        self._fine_tune_epochs = fine_tune_epochs
        self._batch_size = batch_size
        self._model = None
        """:type : keras.models.Sequential"""

    def _pretraining(self, X):
        """
        Pre-train the NN using stacked denoising autoencoders
        :param X: features
        :param y: labels
        :return: encoders
        """
        logger.info(u"Pre-training the denoising autoencoder")

        ae = models.Sequential()

        encoder = containers.Sequential([core.Dropout(0.3), core.Dense(self._input_size, self._layer,
                                                                       activation=self._activation)])
        decoder = containers.Sequential([core.Dense(self._layer, self._input_size, activation=self._activation)])

        ae.add(core.AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

        ae.compile(loss='mean_squared_error', optimizer='sgd')

        ae.fit(X, X, batch_size=self._batch_size, nb_epoch=self._pre_train_epochs)

        return ae.layers[0].encoder

    def _fine_tuning(self, X, y, encoder):
        self._model = models.Sequential()

        logger.info(u"Fine tuning of the neural network (with regularization)")

        self._model.add(encoder)

        self._model.add(core.Dense(
            self._layer, self._classes_amount, activation='softmax', W_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.activity_l2(0.01))
        )

        self._model.compile(loss='categorical_crossentropy', optimizer='sgd')

        self._model.fit(X, y, batch_size=self._batch_size, show_accuracy=True, nb_epoch=self._fine_tune_epochs)

    def fit(self, X, y):
        Y = np_utils.to_categorical(y, self._classes_amount)

        encoder = self._pretraining(X.todense())

        self._fine_tuning(X.todense(), Y, encoder)

    def predict(self, X):
        return self._model.predict_classes(X.todense())

    def predict_proba(self, X):
        return self._model.predict(X.todense())
