# -*- coding: utf-8 -*-


class BaseProcessor(object):
    def __init__(self, corpus, window_size=5):
        self.corpus = corpus
        """:type : dnnwsd.corpus.base.Corpus"""
        self.window_size = window_size
        """:type : int"""
        self.labels = sorted({sentence.sense for sentence in self.corpus})
        """:type : list of unicode"""
        self.dataset = None
        """:type : numpy.ndarray"""
        self.target = None
        """:type : numpy.ndarray"""

    def instances(self):
        raise NotImplementedError
