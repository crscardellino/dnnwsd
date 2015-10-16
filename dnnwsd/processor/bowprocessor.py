# -*- coding: utf-8 -*-

import logging
import numpy as np
from collections import defaultdict
from .base import BaseProcessor
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class BOWProcessor(BaseProcessor):
    def __init__(self, corpus, vocabulary_filter=5, window_size=2):
        super(BOWProcessor, self).__init__(corpus)
        self.vocabulary_filter = vocabulary_filter
        self.window_size = window_size
        self.vocabulary = self._corpus_vocabulary()
        self.vocabulary.sort()

    def _corpus_vocabulary(self):
        logger.info(u"Getting the vocabulary of the corpus from lemma {}".format(self.corpus.lemma).encode("utf-8"))

        vocabulary = defaultdict(int)

        for sentence in self.corpus:
            for word in sentence.predicate_window(self.window_size):
                vocabulary[word.token] += 1

        logger.info(u"Filtering the vocabulary of the corpus from lemma {}".format(self.corpus.lemma).encode("utf-8"))

        return np.array([word for word, count in vocabulary.iteritems() if count >= self.vocabulary_filter])

    def instances(self):
        features = []
        target = []

        for sentence in self.corpus:
            window = np.array([word.token for word in sentence.predicate_window(self.window_size)])  # Get the BoW set
            bow = np.searchsorted(self.vocabulary, window)  # Get the indices of every word in the vocabulary
            features.append(np.bincount(bow).astype(np.int32))  # Counts the amount of words
            target.append(self.labels.index(sentence.sense))

        self.features = np.vstack(features)
        self.target = np.array(target, dtype=np.int32)
