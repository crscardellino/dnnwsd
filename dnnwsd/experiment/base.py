# -*- coding: utf-8 -*-


class BaseExperiment(object):
    def __init__(self, features, target):
        """
        :type features: numpy.ndarray
        :type target: numpy.ndarray
        """
        self.features = features
        self.target = target
