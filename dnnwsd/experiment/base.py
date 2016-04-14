# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import shutil

from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
        self._labels = dataset.labels
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

        # dictionary of results
        if self._validation_ds:
            full_target = np.hstack((
                self._train_ds.target,
                self._test_ds.target,
                self._validation_ds.target
            ))
        else:
            full_target = np.hstack((
                self._train_ds.target,
                self._test_ds.target,
            ))

        self._target_counts = [c[0] for c in Counter(full_target).most_common()]

        self._results = dict(
            train=[],
            train_error=[],
            test=[],
            validation=[],
            train_mcp=[],
            test_mcp=[],
            validation_mcp=[],
            train_lcr=[],
            test_lcr=[],
            validation_lcr=[]
        )

    @property
    def results(self):
        return self._results

    def _add_result(self, y_true, y_pred, dataset):
        assert dataset in {'train', 'test', 'validation'}

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(len(self._labels))
        )
        mcp = precision[self._target_counts[0]]
        lcr = recall[self._target_counts[1:]].mean()

        self._results[dataset].append(accuracy)
        self._results['{}_mcp'.format(dataset)].append(mcp)
        self._results['{}_lcr'.format(dataset)].append(lcr)

    def save_results(self, results_path):
        if os.path.exists(results_path):
            shutil.rmtree(results_path)

        os.makedirs(results_path)

        for dataset in ['train', 'test', 'validation']:
            np.savetxt(
                os.path.join(results_path, dataset), np.array(
                    self._results[dataset], dtype=np.float32
                ), fmt="%.2f"
            )
            np.savetxt(
                os.path.join(results_path, '{}_mcp'.format(dataset)), np.array(
                    self._results['{}_mcp'.format(dataset)], dtype=np.float32
                ), fmt="%.2f"
            )
            np.savetxt(
                os.path.join(results_path, '{}_lcr'.format(dataset)), np.array(
                    self._results['{}_lcr'.format(dataset)], dtype=np.float32
                ), fmt="%.2f"
            )

        np.savetxt(
            os.path.join(results_path, 'train_error'), np.array(self._results['train_error'], dtype=np.float32),
            fmt="%.2f"
        )

    def run(self, results_path):
        raise NotImplementedError
