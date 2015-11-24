# -*- coding: utf-8 -*-

import logging
import numpy as np

from .base import BaseProcessor
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class WordVectorsProcessor(BaseProcessor):
    name = u"Word Vectors Processor"

    def __init__(self, corpus, word2vec_model, window_size=5):
        assert window_size > 0
        super(WordVectorsProcessor, self).__init__(corpus, window_size)

        self.word2vec_model = word2vec_model
        self.vector_size = self.word2vec_model.vector_size

    def _get_window_vector(self, sentence):
        window_vector = []

        for word in sentence.predicate_window(self.window_size):
            tokens_and_lemmas = [s for s in word.tokens_and_lemmas() if s in self.word2vec_model]

            if tokens_and_lemmas:  # If there is an existing combination, take the best one (the first)
                window_vector.append(self.word2vec_model[tokens_and_lemmas[0]])
            else:  # If no possible combination is found, use a zero pad. TODO: What is the best solution?
                window_vector.append(np.zeros(self.vector_size, dtype=np.float32))

        window_vector = np.hstack(window_vector)  # Stack all the vectors in one large vector

        # Padding the window vector in case the predicate is located near the start or end of the sentence
        if sentence.predicate_index - self.window_size < 0:  # Pad to left if the predicate is near to the start
            pad = abs(sentence.predicate_index - self.window_size)
            window_vector = np.hstack((np.zeros(pad * self.vector_size, dtype=np.float32), window_vector))

        if sentence.predicate_index + self.window_size + 1 > len(sentence):
            # Pad to right if the predicate is near to the end
            pad = sentence.predicate_index + self.window_size + 1 - len(sentence)
            window_vector = np.hstack((window_vector, np.zeros(pad * self.vector_size, dtype=np.float32)))

        return window_vector

    def instances(self, force=False):
        if self.dataset and self.target and not force:
            logger.warn(
                u"The corpus dataset and target are already existent and will not be overwritten. " +
                u"To force overwrite use the method with force == True"
            )
            return

        dataset = []
        target = []

        logger.info(u"Getting the dataset and target of window vectors of the corpus from lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        for sentence in self.corpus:
            window_vector = self._get_window_vector(sentence)

            dataset.append(window_vector)
            target.append(self.labels.index(sentence.sense))

        logger.info(u"Dataset and target obtained from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.dataset = np.vstack(dataset)
        self.target = np.array(target, dtype=np.int32)

    def features_dimension(self):
        return self.vector_size * (self.window_size * 2 + 1)


class WordVectorsPoSProcessor(WordVectorsProcessor):
    name = u"Word Vectors with PoS Processor"

    def __init__(self, corpus, word2vec_model, window_size=5):
        super(WordVectorsPoSProcessor, self).__init__(corpus, word2vec_model, window_size)

        self.pos_tags = sorted(set(corpus.pos_tags(self.window_size)))
        self.pos_size = len(self.pos_tags)

    def _get_tag_one_hot_encoding(self, tag):
        one_hot = np.zeros(self.pos_size, dtype=np.float32)
        one_hot[self.pos_tags.index(tag)] = 1.

        return one_hot

    def _get_tags_vector(self, sentence):
        tags_vector = []

        for word in sentence.predicate_window(self.window_size):
            tags_vector.append(self._get_tag_one_hot_encoding(word.tag))

        tags_vector = np.hstack(tags_vector)  # Stack all the vectors in one large vector

        # Padding the window vector in case the predicate is located near the start or end of the sentence
        if sentence.predicate_index - self.window_size < 0:  # Pad to left if the predicate is near to the start
            pad = abs(sentence.predicate_index - self.window_size)
            tags_vector = np.hstack((np.zeros(pad * self.pos_size, dtype=np.float32), tags_vector))

        if sentence.predicate_index + self.window_size + 1 > len(sentence):
            # Pad to right if the predicate is near to the end
            pad = sentence.predicate_index + self.window_size + 1 - len(sentence)
            tags_vector = np.hstack((tags_vector, np.zeros(pad * self.pos_size, dtype=np.float32)))

        return tags_vector

    def instances(self, force=False):
        if self.dataset and self.target and not force:
            logger.warn(
                u"The corpus dataset and target are already existent and will not be overwritten. " +
                u"To force overwrite use the method with force == True"
            )
            return

        dataset = []
        target = []

        logger.info(u"Getting the dataset and target of window vectors of the corpus from lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        for sentence in self.corpus:
            window_vector = self._get_window_vector(sentence)
            tags_vector = self._get_tags_vector(sentence)

            dataset.append(np.hstack((window_vector, tags_vector)))
            target.append(self.labels.index(sentence.sense))

        logger.info(u"Dataset and target obtained from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.dataset = np.vstack(dataset)
        self.target = np.array(target, dtype=np.int32)

    def features_dimension(self):
        return (self.vector_size + self.pos_size) * (self.window_size * 2 + 1)


class SemiSupervisedWordVectorsProcessor(WordVectorsProcessor):
    name = u"Semi Supervised Word Vectors Processor"

    def __init__(self, corpus, unannotated_corpus, word2vec_model, window_size=5, sample_ratio=1.):
        assert 0. < sample_ratio <= 1.
        super(SemiSupervisedWordVectorsProcessor, self).__init__(corpus, word2vec_model, window_size)

        self.unannotated_corpus = unannotated_corpus
        """:type : dnnwsd.corpus.unannotated.UnannotatedCorpus"""
        self.unannotated_dataset = None
        """:type : numpy.ndarray"""
        self.automatic_dataset = np.array([], dtype=np.float32)
        # All the automatically annotated data will be moved here
        """:type : numpy.ndarray"""
        self.automatic_target = np.array([], dtype=np.int32)  # This will store the automatically annotated targets
        """:type : numpy.ndarray"""
        self._sample_ratio = sample_ratio

    def instances(self, force=False):
        super(SemiSupervisedWordVectorsProcessor, self).instances(force)

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
            window_vector = self._get_window_vector(sentence)

            untagged_dataset.append(window_vector)

        logger.info(u"Dataset and target obtained from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.unannotated_dataset = np.vstack(untagged_dataset)
        self.automatic_dataset = self.automatic_dataset.reshape(0, self.unannotated_dataset.shape[1])

    def tag_slice(self, slice_range, target):
        self.automatic_dataset, self.unannotated_dataset = (
            np.vstack((self.automatic_dataset, self.unannotated_dataset[slice_range])),
            np.delete(self.unannotated_dataset, slice_range, axis=0)
        )

        self.automatic_target = np.hstack((self.automatic_target, target))

    def untagged_corpus_size(self):
        return int(len(self.unannotated_corpus) * self._sample_ratio)

    def untagged_corpus_proportion(self):
        return self.unannotated_dataset.shape[0], self.automatic_dataset.shape[0]


class SemiSupervisedWordVectorsPoSProcessor(WordVectorsPoSProcessor):
    name = u"Semi-Supervised Word Vectors with PoS Processor"

    def __init__(self, corpus, word2vec_model, pos_tags, window_size=5, sample_ratio=1.):
        super(SemiSupervisedWordVectorsPoSProcessor, self).__init__(corpus, word2vec_model, window_size)

        self.pos_tags = pos_tags
        self.pos_size = len(self.pos_tags)

        self.unannotated_corpus = unannotated_corpus
        """:type : dnnwsd.corpus.unannotated.UnannotatedCorpus"""
        self.unannotated_dataset = None
        """:type : numpy.ndarray"""
        self.automatic_dataset = np.array([], dtype=np.float32)
        # All the automatically annotated data will be moved here
        """:type : numpy.ndarray"""
        self.automatic_target = np.array([], dtype=np.int32)  # This will store the automatically annotated targets
        """:type : numpy.ndarray"""
        self._sample_ratio = sample_ratio

    def instances(self, force=False):
        super(SemiSupervisedWordVectorsPoSProcessor, self).instances(force)

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
            window_vector = self._get_window_vector(sentence)
            tags_vector = self._get_tags_vector(sentence)

            untagged_dataset.append(np.hstack((window_vector, tags_vector)))

        logger.info(u"Dataset and target obtained from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.unannotated_dataset = np.vstack(untagged_dataset)
        self.automatic_dataset = self.automatic_dataset.reshape(0, self.unannotated_dataset.shape[1])

    def tag_slice(self, slice_range, target):
        self.automatic_dataset, self.unannotated_dataset = (
            np.vstack((self.automatic_dataset, self.unannotated_dataset[slice_range])),
            np.delete(self.unannotated_dataset, slice_range, axis=0)
        )

        self.automatic_target = np.hstack((self.automatic_target, target))

    def untagged_corpus_size(self):
        return int(len(self.unannotated_corpus) * self._sample_ratio)

    def untagged_corpus_proportion(self):
        return self.unannotated_dataset.shape[0], self.automatic_dataset.shape[0]
