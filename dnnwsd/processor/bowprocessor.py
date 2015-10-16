# -*- coding: utf-8 -*-

import logging
import numpy as np
from collections import defaultdict
from .base import BaseProcessor
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class BoWProcessor(BaseProcessor):
    def __init__(self, corpus, vocabulary_filter=5, window_size=2):
        super(BoWProcessor, self).__init__(corpus)
        self.vocabulary_filter = vocabulary_filter
        self.window_size = window_size
        self._get_corpus_features()

    def _get_corpus_features(self):
        logger.info(u"Getting the vocabulary of the corpus from lemma {}".format(self.corpus.lemma).encode("utf-8"))

        vocabulary = defaultdict(int)

        for sentence in self.corpus:
            for word in sentence.predicate_window(self.window_size):
                vocabulary[word.token] += 1

        logger.info(u"Filtering the vocabulary of the corpus from lemma {}".format(self.corpus.lemma).encode("utf-8"))

        self.features = \
            np.sort(np.array([word for word, count in vocabulary.iteritems() if count >= self.vocabulary_filter]))

    def instances(self):
        dataset = []
        target = []

        for sentence in self.corpus:
            window = np.array([word.token for word in sentence.predicate_window(self.window_size)])  # Get the BoW set
            bow = np.searchsorted(self.features, window)  # Get the indices of every word in the vocabulary
            dataset.append(np.bincount(bow).astype(np.int32))  # Counts the amount of words
            target.append(self.labels.index(sentence.sense))

        self.dataset = np.vstack(dataset)
        self.target = np.array(target, dtype=np.int32)


class BoPoSProcessor(BoWProcessor):
    def __init__(self, corpus, vocabulary_filter=5, window_size=2, pos_filter=0):
        self.pos_filter = pos_filter
        super(BoPoSProcessor, self).__init__(corpus, vocabulary_filter, window_size)

    def _get_corpus_features(self):
        super(BoPoSProcessor, self)._get_corpus_features()

        logger.info(u"Getting the part of speech tags of the corpus from lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        postags = defaultdict(int)

        for sentence in self.corpus:
            for word in sentence.predicate_window(self.window_size):
                postags[word.tag] += 1

        logger.info(u"Filtering the part of speech tags of the corpus from lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.features = \
            np.hstack((
                self.features,
                np.sort(np.array([pos for pos, count in postags.iteritems() if count >= self.pos_filter]))
            ))

    def instances(self):
        dataset = []
        target = []

        for sentence in self.corpus:
            window = np.array([word.token for word in sentence.predicate_window(self.window_size)])
            window = np.hstack((window, [word.tag for word in sentence.predicate_window(self.window_size)]))
            bow = np.searchsorted(self.features, window)
            dataset.append(np.bincount(bow).astype(np.int32))
            target.append(self.labels.index(sentence.sense))

        self.dataset = np.vstack(dataset)
        self.target = np.array(target, dtype=np.int32)
