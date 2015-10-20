# -*- coding: utf-8 -*-

import logging
from .base import BaseModel
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SKLearnWrapperModel(BaseModel):
    """
    Class to wrap a scikit-learn model.
    """
    def __init__(self, sklearn_model_class, sklearn_model_parameters):
        self._model = sklearn_model_class(**sklearn_model_parameters)

    def fit(self, X, y):
        logger.info("Fitting the classifier {}".format(self._model.__class__.__name__))
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
