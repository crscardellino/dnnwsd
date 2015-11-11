# -*- coding: utf-8 -*-

TRAIN_RATIO = 0.8


class Experiment(object):
    def __init__(self, processor, model):
        self._processor = processor
        """:type : dnnwsd.processor.base.BaseProcessor"""
        self._model = model
        """:type : dnnwsd.model.base.BaseModel"""

    def split_dataset(self):
        raise NotImplementedError
