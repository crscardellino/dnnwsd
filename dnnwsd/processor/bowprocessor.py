# -*- coding: utf-8 -*-

import logging
import numpy as np
from collections import defaultdict
from scipy import sparse
from .base import BaseProcessor
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class BoWProcessor(BaseProcessor):
    def __init__(self, corpus, vocabulary_filter=2, window_size=5):
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

    def _get_instance_data(self, sentence):
        return np.array([word.token for word in sentence.predicate_window(self.window_size)])

    def instances(self):
        dataset = []
        target = []

        for sentence in self.corpus:
            # Get the data (depends on the processor's class)
            instance_features = self._get_instance_data(sentence)

            # Get the indices of every word in the vocabulary (that was not filtered)
            instance_features = np.searchsorted(
                self.features, instance_features[np.in1d(instance_features, self.features)]
            )

            # Counts the amounts
            instance_features = np.bincount(instance_features).astype(np.int32)

            # Resize to match the features shape
            instance_features.resize(self.features.shape)

            dataset.append(instance_features)
            target.append(self.labels.index(sentence.sense))

        self.dataset = sparse.csr_matrix(np.vstack(dataset))
        self.target = np.array(target, dtype=np.int32)


class BoPoSProcessor(BoWProcessor):
    def __init__(self, *args, **kwargs):
        self.pos_filter = kwargs.pop('pos_filter', 2)
        super(BoPoSProcessor, self).__init__(*args, **kwargs)

    def _get_corpus_features(self):
        super(BoPoSProcessor, self)._get_corpus_features()

        logger.info(u"Getting the part of speech tags of the corpus from lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        postags = defaultdict(int)

        for sentence in self.corpus:
            for widx, word in enumerate(sentence.predicate_window(self.window_size)):
                postags[self._get_tag_literal(word, widx)] += 1

        logger.info(u"Filtering the part of speech tags of the corpus from lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.features = \
            np.hstack((
                self.features,
                np.sort(np.array([pos for pos, count in postags.iteritems() if count >= self.pos_filter]))
            ))

    def _get_tag_literal(self, *args):
        return args[0].tag

    def _get_instance_data(self, sentence):
        instance_data = np.array([word.token for word in sentence.predicate_window(self.window_size)])

        tags_literal = map(
            lambda (widx, word): self._get_tag_literal(word, widx),
            enumerate(sentence.predicate_window(self.window_size))
        )

        instance_data = np.hstack((instance_data, tags_literal))

        return instance_data


class PoSProcessor(BoPoSProcessor):
    def _get_tag_literal(self, *args):
        return u"{}{:+d}".format(args[0].tag, args[1]-self.window_size)
