# -*- coding: utf-8 -*-

import logging

from ..utils.dataset import DataSets
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.8


class Experiment(object):
    def __init__(self, processor, model):
        self._processor = processor
        """:type : dnnwsd.processor.base.BaseProcessor"""
        self._model = model
        """:type : dnnwsd.model.base.BaseModel"""

    def split_dataset(self):
        raise NotImplementedError


class NeuralNetworkExperiment(object):
    def __init__(self, dataset_path_or_instance, epochs, starter_learning_rate,
                 train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1):

        if type(dataset_path_or_instance) == str:
            dataset = DataSets(dataset_path_or_instance, train_ratio, test_ratio, validation_ratio)
        elif type(dataset_path_or_instance) == DataSets:
            dataset = dataset_path_or_instance
        else:
            raise Exception("The provided dataset is not valid")

        self._lemma = dataset.lemma
        self._train_ds = dataset.train_ds.annotated_ds
        self._test_ds = dataset.test_ds
        self._validation_ds = dataset.validation_ds

        logger.info(u"Dataset for lemma {} loaded.".format(self._lemma))

        self._input_size = self._train_ds.vector_length
        self._output_size = self._train_ds.labels_count

        self._num_examples = self._train_ds.data_count
        self._batch_size = self._train_ds.data_count
        self._epochs = epochs

        self._starter_learning_rate = starter_learning_rate

        self._results = dict(
            train=[],
            train_error=[],
            test=[],
            validation=[]
        )

    @property
    def results(self):
        return self._results

    def run(self, results_path):
        raise NotImplementedError
