# -*- coding: utf-8 -*-

import numpy as np


class BaseProcessor(object):
    name = u"Base Processor"

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

    def instances(self, force=False):
        raise NotImplementedError

    def load_data(self, load_path):
        data = np.load(load_path)

        self.dataset = data['dataset']
        self.target = data['target']

    def save_data(self, save_path):
        np.savez(save_path, dataset=self.dataset, target=self.target)
