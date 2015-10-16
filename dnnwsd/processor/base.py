# -*- coding: utf-8 -*-


class BaseProcessor(object):
    def __init__(self, corpus):
        """
        :type corpus: ddnwsd.corpus.base.Corpus
        """

        self.corpus = corpus
        self.labels = sorted({sentence.sense for sentence in self.corpus})
        """:type : list of unicode"""
        self.features = None
        """:type : numpy.ndarrray"""
        self.dataset = None
        """:type : numpy.ndarray"""
        self.target = None
        """:type : numpy.ndarray"""

    def _get_corpus_features(self):
        raise NotImplementedError

    def instances(self):
        raise NotImplementedError
