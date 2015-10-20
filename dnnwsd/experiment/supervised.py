# -*- coding: utf -8 -*-

import logging
from sklearn.cross_validation import StratifiedShuffleSplit
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1


class SupervisedExperiment(object):
    def __init__(self, corpus, processor_class, processor_parameters, model_class, model_parameters):
        self._corpus = corpus
        """:type : dnnwsd.corpus.base.Corpus"""

        self._processor_class = processor_class
        self._processor_parameters = processor_parameters

        self._model_class = model_class
        self._model_parameters = model_parameters

    @staticmethod
    def split_dataset(processor):
        dataset = {}

        split = StratifiedShuffleSplit(processor.target, n_iter=1, train_size=TRAIN_RATIO, test_size=None)

        for train_index, test_index in split:
            dataset['X_train'], dataset['X_test'] = processor.dataset[train_index], processor.dataset[test_index]
            dataset['y_train'], dataset['y_test'] = processor.target[train_index], processor.target[test_index]

        return dataset

    def run(self, results):
        """
        :type results: dnnwsd.utils.results.ResultsHandler
        """
        logger.info(u"Running supervised experiment from the corpus of lemma {}".format(self._corpus.lemma))

        logger.info(u"Setting up the corpus processor")

        processor = self._processor_class(self._corpus, **self._processor_parameters)
        """:type : dnnwsd.processor.base.BaseProcessor"""

        processor.instances()

        logger.info(u"Setting up the experiment model of class {}".format(self._model_class.__name__))

        model = self._model_class(**self._model_parameters)
        """:type : dnnwsd.model.base.BaseModel"""

        logger.info(u"Splitting the dataset")

        dataset_split = self.split_dataset(processor)

        logger.info(u"Fitting the classifier")

        model.fit(dataset_split['X_train'], dataset_split['y_train'])

        logger.info(u"Getting results from the classifier")

        results.add_result(dataset_split['y_test'], model.predict(dataset_split['X_test']))

        logger.info(u"Finished supervised experiment from the corpus of lemma {}".format(self._corpus.lemma))
