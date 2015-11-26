# -*- coding: utf-8 -*-

import cPickle
import logging
import numpy as np
import scipy.sparse as sp

from collections import defaultdict
from scipy import sparse
from .base import BaseProcessor

from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class BoWProcessor(BaseProcessor):
    name = u"Bag of Words Processor"

    def __init__(self, corpus, vocabulary_filter=3, window_size=5):
        super(BoWProcessor, self).__init__(corpus, window_size)
        self._vocabulary_filter = vocabulary_filter
        self._features = None
        """:type : numpy.ndarrray"""

        self._get_corpus_features()

    def _get_corpus_features(self):
        logger.info(u"Getting the vocabulary from the corpus of lemma {}".format(self.corpus.lemma).encode("utf-8"))

        vocabulary = defaultdict(int)

        for sentence in self.corpus:
            for word in sentence.predicate_window(self.window_size):
                vocabulary[word.token] += 1

        logger.info(u"Filtering the vocabulary from the corpus of lemma {}".format(self.corpus.lemma).encode("utf-8"))

        self._features = \
            np.sort(np.array([word for word, count in vocabulary.iteritems() if count >= self._vocabulary_filter]))

    def _get_instance_data(self, sentence):
        return np.array([word.token for word in sentence.predicate_window(self.window_size)])

    def instances(self, force=False):
        if self.dataset and self.target and not force:
            logger.warn(
                u"The corpus dataset and target are already existent and will not be overwritten. " +
                u"To force overwrite use the method with force == True"
            )
            return

        dataset = []
        target = []

        logger.info(u"Getting corpus dataset and target from sentences from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        for sentence in self.corpus:
            # Get the data (depends on the processor's class)
            instance_features = self._get_instance_data(sentence)

            # Get the indices of every word in the vocabulary (that was not filtered)
            instance_features = np.searchsorted(
                self._features, instance_features[np.in1d(instance_features, self._features)]
            )

            # Counts the amounts
            instance_features = np.bincount(instance_features).astype(np.int32)

            # Resize to match the features shape
            instance_features.resize(self._features.shape)

            dataset.append(instance_features)
            target.append(self.labels.index(sentence.sense))

        logger.info(u"Dataset and target obtainded from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.dataset = sparse.csr_matrix(np.vstack(dataset))
        self.target = np.array(target, dtype=np.int32)

    def features_dimension(self):
        return self._features.shape[0]


class BoPoSProcessor(BoWProcessor):
    name = u"Bag of Part-of-Speech Processor"

    def __init__(self, *args, **kwargs):
        self.pos_filter = kwargs.pop('pos_filter', 2)
        super(BoPoSProcessor, self).__init__(*args, **kwargs)

    def _get_corpus_features(self):
        super(BoPoSProcessor, self)._get_corpus_features()

        logger.info(u"Getting the part of speech tags from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        postags = defaultdict(int)

        for sentence in self.corpus:
            for widx, word in enumerate(sentence.predicate_window(self.window_size)):
                postags[self._get_tag_literal(word, widx)] += 1

        logger.info(u"Filtering the part of speech tags from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self._features = \
            np.hstack((
                self._features,
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
    name = u"Part-of-Speech with Positions Processor"

    def _get_tag_literal(self, *args):
        return u"{}{:+d}".format(args[0].tag, args[1]-self.window_size)


class SemiSupervisedBoWProcessor(BoWProcessor):
    name = u"Semi-Supervised Bag-of-Words Processor"

    def __init__(self, corpus, unannotated_corpus, features_path, vocabulary_filter=3, window_size=5, sample_ratio=1.):
        assert 0. < sample_ratio <= 1.
        self._features_path = features_path
        super(SemiSupervisedBoWProcessor, self).__init__(corpus, vocabulary_filter, window_size)

        self.unannotated_corpus = unannotated_corpus
        """:type : dnnwsd.corpus.unannotated.UnannotatedCorpus"""
        self.unannotated_dataset = None
        """:type : numpy.ndarray"""
        self.automatic_dataset = sp.csr_matrix((0, 0), dtype=np.float32)
        # All the automatically annotated data will be moved here
        """:type : numpy.ndarray"""
        self.automatic_target = np.array([], dtype=np.int32)  # This will store the automatically annotated targets
        """:type : numpy.ndarray"""
        self._sample_ratio = sample_ratio

    def _get_corpus_features(self):
        with open(self._features_path, 'rb') as f:
            features = dict(cPickle.load(f))

        self._features = \
            np.sort(np.array([word for word, count in features.iteritems() if count >= self._vocabulary_filter]))

    def _get_instance_data(self, sentence):
        return np.array([word.token for word in sentence.predicate_window(self.window_size)])

    def instances(self, force=False):
        super(SemiSupervisedBoWProcessor, self).instances(force)

        untagged_dataset = []

        logger.info(u"Getting the untagged dataset and target of window vectors of the corpus from lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        if self._sample_ratio < 1.:
            sample_size = int(len(self.unannotated_corpus) * self._sample_ratio)
            corpus_choices = set(np.random.choice(len(self.unannotated_corpus), size=sample_size, replace=False))
            corpus_sentences = (s for i, s in enumerate(self.unannotated_corpus) if i in corpus_choices)
        else:
            corpus_sentences = (s for s in self.unannotated_corpus)

        for sentence in corpus_sentences:
            # Get the data (depends on the processor's class)
            instance_features = self._get_instance_data(sentence)

            # Get the indices of every word in the vocabulary (that was not filtered)
            instance_features = np.searchsorted(
                self._features, instance_features[np.in1d(instance_features, self._features)]
            )

            # Counts the amounts
            instance_features = np.bincount(instance_features).astype(np.int32)

            # Resize to match the features shape
            instance_features.resize(self._features.shape)

            untagged_dataset.append(instance_features)

        logger.info(u"Dataset and target obtained from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.unannotated_dataset = sparse.csr_matrix(np.vstack(untagged_dataset))
        self.automatic_dataset = sp.csr_matrix((0, self.unannotated_dataset.shape[1]))

    def tag_slice(self, slice_range, target):
        self.automatic_dataset = sp.vstack((self.automatic_dataset, self.unannotated_dataset[slice_range]))

        # Delete the slice from the unannotated_dataset
        mask = np.ones(self.unannotated_dataset.shape[0], dtype=bool)
        mask[slice_range] = False
        self.unannotated_dataset = self.unannotated_dataset[mask]

        self.automatic_target = np.hstack((self.automatic_target, target))

    def untagged_corpus_size(self):
        return int(len(self.unannotated_corpus) * self._sample_ratio)

    def untagged_corpus_proportion(self):
        return self.unannotated_dataset.shape[0], self.automatic_dataset.shape[0]
