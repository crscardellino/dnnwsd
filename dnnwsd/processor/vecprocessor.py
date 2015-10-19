# -*- coding: utf-8 -*-

import gensim
import logging
import numpy as np
from scipy import sparse
from .base import BaseProcessor
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class WordVectorsProcessor(BaseProcessor):
    def __init__(self, corpus, word2vec_model_path, window_size=5):
        assert window_size > 0
        super(WordVectorsProcessor, self).__init__(corpus)

        logger.info("Loading Wor2Vec model from {}".format(word2vec_model_path))
        self.word2vec_model = gensim.models.Word2Vec.load_word2vec_format(word2vec_model_path, binary=True)
        logger.info("Finished loading Word2Vec model")

        self.window_size = window_size
        self.vector_size = self.word2vec_model.vector_size
        self.features_dimension = self.vector_size * (self.window_size * 2 + 1)

    def _get_window_vector(self, sentence):
        window_vector = []

        for word in sentence.predicate_window(self.window_size):
            tokens_and_lemmas = [s for s in word.tokens_and_lemmas() if s in self.word2vec_model]

            if tokens_and_lemmas:  # If there is an existing combination, take the best one (the first)
                window_vector.append(self.word2vec_model[tokens_and_lemmas[0]])
            else:  # If no possible combination is found, use a zero pad. TODO: What is the best solution?
                window_vector.append(np.zeros(self.vector_size, dtype='float32'))

        window_vector = np.hstack(window_vector)  # Stack all the vector in one large vector

        # Padding the window vector in case the predicate is located near the start or end of the sentence
        if sentence.predicate_index - self.window_size < 0:  # Pad to left if the predicate is near to the start
            pad = abs(sentence.predicate_index - self.window_size)
            window_vector = np.hstack((np.zeros(pad * self.vector_size, dtype='float32'), window_vector))

        if sentence.predicate_index + self.window_size + 1 > len(sentence):
            # Pad to right if the predicate is near to the end
            pad = sentence.predicate_index + self.window_size + 1 - len(sentence)
            window_vector = np.hstack((window_vector, np.zeros(pad * self.vector_size, dtype='float32')))

        return window_vector

    def instances(self):
        dataset = []
        target = []

        logger.info(u"Getting the dataset and target of window vectors of the corpus from lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        for sentence in self.corpus:
            window_vector = self._get_window_vector(sentence)

            dataset.append(window_vector)
            target.append(self.labels.index(sentence.sense))

        logger.info(u"Dataset and target obtainded from the corpus of lemma {}"
                    .format(self.corpus.lemma).encode("utf-8"))

        self.dataset = sparse.csr_matrix(np.vstack(dataset))
        self.target = np.array(target, dtype=np.int32)

