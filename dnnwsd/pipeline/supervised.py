# -*- coding: utf-8 -*-

import cPickle as pickle
import logging
import os

from sklearn import ensemble, linear_model, tree

from ..corpus import sensem
from ..experiment import results, supervised
from ..model import mlp, mfl
from ..processor import bowprocessor, vecprocessor
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SupervisedPipeline(object):
    processors_map = {
        'bow': bowprocessor.BoWProcessor,
        'bopos': bowprocessor.BoPoSProcessor,
        'pos': bowprocessor.PoSProcessor,
        'wordvec': vecprocessor.WordVectorsProcessor,
        'wordvecpos': vecprocessor.WordVectorsPoSProcessor
    }

    models_map = {
        'decisiontree': tree.DecisionTreeClassifier,
        'logreg': linear_model.LogisticRegression,
        'mfl': mfl.MostFrequentLabel,
        'mlp': mlp.MultiLayerPerceptron,
        'randomforest': ensemble.RandomForestClassifier
    }

    def __init__(self, corpus_directory, results_directory, experiment_set, **kwargs):
        self._corpus_iterator = sensem.SenSemCorpusDirectoryIterator(
            corpus_directory, kwargs.pop('sense_filter', 3)
        )
        self._results_directory = results_directory
        self._experiment_set = experiment_set
        # List of 4-tuples, each defining an experiment.
        # (processor, processor_parameters, model, model_parameters)
        self._iterations = kwargs.pop('iterations', 5)
        self._save_corpus_path = kwargs.pop('save_corpus_path', "")
        self._save_datasets_path = kwargs.pop('save_datasets_path', "")
        self._load_datasets_path = kwargs.pop('load_datasets_path', "")

    def _run_for_corpus(self, corpus):
        """
        :param corpus: ddnwsd.corpus.sensem.SenSemCorpus
        """

        lemma_index = self._corpus_iterator.verbs.index(corpus.lemma)

        experiments_dir = os.path.join(self._results_directory, "{:03d}".format(lemma_index))

        for (pkey, pparam, mkey, mparam) in self._experiment_set:
            if mkey == 'mlp':
                experiment_name = "{}_{}_{}_{}".format(
                    pkey, mkey, mparam.get('layers'), mparam.get('pre_train_epochs', 0)
                )
            else:
                experiment_name = "{}_{}".format(pkey, mkey)

            results_save_path = os.path.join(experiments_dir, experiment_name)
            os.makedirs(results_save_path)

            processor = self.processors_map[pkey](corpus, **pparam)
            """:type : dnnwsd.processor.base.BaseProcessor"""
            dataset_path = os.path.join(self._load_datasets_path, "{:03d}_{}.npz".format(lemma_index, pkey))

            if os.path.isfile(dataset_path):
                processor.load_data(dataset_path)
            else:
                processor.instances()

            if self._save_datasets_path:
                dataset_path = os.path.join(self._save_datasets_path, "{:03d}_{}.npz".format(lemma_index, pkey))
                if not os.path.isfile(dataset_path):  # Check if wasn't already created
                    processor.save_data(dataset_path)

            if mkey == 'mfl':
                mparam['labels'] = processor.labels

            if mkey == 'mlp':
                mparam['input_dim'] = processor.features_dimension()
                mparam['output_dim'] = len(processor.labels)

            model = self.models_map[mkey](**mparam)
            """:type : dnnwsd.models.base.BaseModel"""

            logger.info(u"Running experiments for {} and model {}".format(processor.name, model.__class__.__name__))

            results_handler = results.ResultsHandler(results_save_path, processor.labels, processor.target)

            experiment = supervised.SupervisedExperiment(processor, model, kfolds=self._iterations)

            experiment.run(results_handler)

            results_handler.save_results()

    def run(self):
        logger.info(u"Running experiments pipeline for whole corpus")

        for corpus in self._corpus_iterator:
            if not corpus.has_multiple_senses() or corpus.lemma == u"estar":
                logger.info(u"Skipping experiments pipeline for lemma {}.".format(corpus.lemma) +
                            u"The corpus doesn't have enough senses")
                continue

            logger.info(u"Running experiments pipeline for lemma {}".format(corpus.lemma))

            if self._save_corpus_path:  # Save the corpus in the path
                lemma_index = self._corpus_iterator.verbs.index(corpus.lemma)
                with open(os.path.join(self._save_corpus_path, "{:03d}.p".format(lemma_index)), "wb") as f:
                    logger.info(u"Saving corpus binary file in {}".format(f.name))
                    pickle.dump(corpus, f)

            self._run_for_corpus(corpus)

            logger.info(u"Finished experiments pipeline for lemma {}".format(corpus.lemma))
