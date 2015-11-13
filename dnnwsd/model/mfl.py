# -*- coding: utf-8 -*-

import numpy as np

from .base import BaseModel


class MostFrequentLabel(BaseModel):
    def __init__(self, labels):
        self.labels = sorted(labels)
        self.predicted_label = 0

    def fit(self, X, y):
        labels_amount = np.zeros(len(self.labels), dtype=y.dtype)

        for label in y:
            labels_amount[label] += 1

        self.predicted_label = np.argmax(labels_amount)

    def predict(self, X):
        return np.array([self.predicted_label for x in xrange(X.shape[0])])

    def predict_proba(self, X):
        probabilities = []

        for x in xrange(X.shape[0]):
            prob_array = np.zeros(X.shape[1], dtype=X.dtype)
            prob_array[self.predicted_label] = 1.
            probabilities.append(prob_array)

        return np.array(probabilities)
