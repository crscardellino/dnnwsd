# -*- coding: utf -8 -*-

import logging

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit

from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1


class SupervisedExperiment(object):
    def __init__(self, processor, model, kfolds=0):
        self._processor = processor
        """:type : dnnwsd.processor.base.BaseProcessor"""
        self._model = model
        """:type : dnnwsd.model.base.BaseModel"""
        self._kfolds = kfolds

    def split_dataset(self):
        if self._kfolds > 0:
            dataset_split = StratifiedKFold(
                self._processor.target, n_folds=self._kfolds, shuffle=True
            )
        else:
            dataset_split = StratifiedShuffleSplit(
                self._processor.target, n_iter=1, train_size=TRAIN_RATIO, test_size=None
            )

        return dataset_split

    def run(self, results_handler):
        """
        :type results_handler: dnnwsd.utils.results.ResultsHandler
        """
        logger.info(u"Splitting the dataset")

        dataset_split = self.split_dataset()

        if self._kfolds > 0:
            logger.info(u"Running {}-fold cross-validation on the dataset".format(self._kfolds))

        for fold_idx, (train_index, test_index) in enumerate(dataset_split):
            if self._kfolds > 0:
                logger.info(u"Running fold {}".format(fold_idx))

            dataset = {
                'X_train': self._processor.dataset[train_index],
                'y_train': self._processor.target[train_index],
                'X_test': self._processor.dataset[test_index],
                'y_test': self._processor.target[test_index]
            }

            logger.info(u"Fitting the classifier")

            self._model.fit(dataset['X_train'], dataset['y_train'])

            logger.info(u"Getting results from the classifier")

            results_handler.add_result(dataset['y_test'], self._model.predict(dataset['X_test']))
