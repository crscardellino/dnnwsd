# -*- coding: utf-8 -*-

import cPickle as pickle
import logging
import os

from sklearn import linear_model

from ..corpus import sensem
from ..experiment import results, supervised
from ..model import autoencoder
from ..processor import bowprocessor, vecprocessor
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SupervisedPipeline(object):
    processors_map = {
        'bow': bowprocessor.BoWProcessor,
        'bopos': bowprocessor.BoPoSProcessor,
        'pos': bowprocessor.PoSProcessor,
        'wordvec': vecprocessor.WordVectorsProcessor
    }

    models_map = {
        'logreg': linear_model.LogisticRegression,
        'autoencoder': autoencoder.DenoisingAutoencoder
    }

    def __init__(self, corpus_directory, results_directory, **kwargs):
        self._corpus_iterator = sensem.SenSemCorpusDirectoryIterator(
            corpus_directory, kwargs.pop('sense_filter', 3)
        )
        self._results_directory = results_directory
        self._iterations = kwargs.pop('iterations', 5)
        self._experiment_set = kwargs.pop('experiment_set', [])
        # List of 4-tuples, each defining an experiment.
        # (processor, processor_parameters, model, model_parameters)
        self._save_corpus_path = kwargs.pop('save_corpus_path', u"")
        self._save_datasets_path = kwargs.pop('save_datasets_path', u"")
        self._load_datasets_path = kwargs.pop('load_datasets_path', u"")

    def _run_for_corpus(self, corpus):
        """
        :param corpus: ddnwsd.corpus.sensem.SenSemCorpus
        """

        experiments_dir = os.path.join(self._results_directory, corpus.lemma)

        for (pkey, pparam, mkey, mparam) in self._experiment_set:
            experiment_name = u"{}_{}".format(pkey, mkey)
            results_save_path = os.path.join(experiments_dir, experiment_name)
            os.makedirs(results_save_path)

            processor = self.processors_map[pkey](corpus, **pparam)
            """:type : dnnwsd.processor.base.BaseProcessor"""
            dataset_path = os.path.join(self._load_datasets_path, u"{}_{}.npz".format(corpus.lemma, pkey))

            if os.path.isfile(dataset_path):
                processor.load_data(dataset_path)
            else:
                processor.instances()

            if self._save_datasets_path:
                dataset_path = os.path.join(self._save_datasets_path, u"{}_{}.npz".format(corpus.lemma, pkey))
                if not os.path.isfile(dataset_path):  # Check if wasn't already created
                    processor.save_data(dataset_path)

            if mkey == 'autoencoder':
                mparam['classes_amount'] = len(processor.labels)
                mparam['input_size'] = processor.features_dimension()

            model = self.models_map[mkey](**mparam)
            """:type : dnnwsd.models.base.BaseModel"""

            logger.info(u"Running experiments for {} and model {}".format(processor.name, model.__class__.__name__))

            results_handler = results.ResultsHandler(experiment_name, results_save_path, processor.labels)

            experiment = supervised.SupervisedExperiment(processor, model, kfolds=self._iterations)

            experiment.run(results_handler)

            results_handler.save_results()

    def run(self):
        logger.info(u"Running experiments pipeline for whole corpus")

        for corpus in self._corpus_iterator:
            if not corpus.has_multiple_senses():
                logger.info(u"Skipping experiments pipeline for lemma {}.".format(corpus.lemma) +
                            u"The corpus doesn't have enough senses")
                continue

            logger.info(u"Running experiments pipeline for lemma {}".format(corpus.lemma))

            if self._save_corpus_path:  # Save the corpus in the path
                with open(os.path.join(self._save_corpus_path, u"{}.p".format(corpus.lemma)), "wb") as f:
                    logger.info(u"Saving corpus binary file in {}".format(f.name))
                    pickle.dump(corpus, f)

            self._run_for_corpus(corpus)

            logger.info(u"Finished experiments pipeline for lemma {}".format(corpus.lemma))
