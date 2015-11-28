# -*- coding: utf-8 -*-

import logging
import os

from sklearn import linear_model

from ..corpus import sensem, unannotated
from ..experiment import results, semisupervised
from ..model import mlp
from ..processor import bowprocessor, vecprocessor
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SemiSupervisedPipeline(object):
    processors_map = {
        'bow': bowprocessor.SemiSupervisedBoWProcessor,
        'wordvec': vecprocessor.SemiSupervisedWordVectorsProcessor,
        'wordvecpos': vecprocessor.SemiSupervisedWordVectorsPoSProcessor
    }

    models_map = {
        'logreg': linear_model.LogisticRegression,
        'mlp': mlp.MultiLayerPerceptron
    }

    def __init__(self, corpus_directory, unannotated_corpus_directory, results_directory,
                 experiment_set, features_path, **kwargs):
        self._corpus_iterator = sensem.SenSemCorpusDirectoryIterator(
            corpus_directory, kwargs.pop('sense_filter', 3)
        )
        self._unannotated_corpus_iterator = unannotated.UnannotatedCorpusDirectoryIterator(
            unannotated_corpus_directory
        )
        self._results_directory = results_directory
        self._experiment_set = experiment_set
        # List of 4-tuples, each defining an experiment.
        # (processor, processor_parameters, model, model_parameters)
        self._features_path = features_path
        self._confidence_threshold = kwargs.pop("confidence_threshold", 0.99)
        self._minimum_instances = kwargs.pop("minimum_instances", None)
        self._max_iterations = kwargs.pop("max_iterations", 100)
        self._evaluation_size = kwargs.pop("evaluation_size", 10)
        self._starting_lemma = kwargs.pop("starting_lemma", 0)

    def _run_for_corpus(self, annotated_corpus, unannotated_corpus, corpus_index):
        """
        :param annotated_corpus: ddnwsd.corpus.sensem.SenSemCorpus
        :param unannotated_corpus: ddnwsd.corpus.unannotated.UnannotatedCorpus
        """

        lemma_index = self._corpus_iterator.verbs.index(annotated_corpus.lemma)

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

            if pkey == 'bow':
                pparam['features_path'] = os.path.join(self._features_path, "{:03d}.p".format(corpus_index))

            processor = self.processors_map[pkey](annotated_corpus, unannotated_corpus, **pparam)
            """:type : dnnwsd.processor.base.BaseProcessor"""
            processor.instances()

            if mkey == 'mlp':
                mparam['input_dim'] = processor.features_dimension()
                mparam['output_dim'] = len(processor.labels)

            model = self.models_map[mkey](**mparam)
            """:type : dnnwsd.models.base.BaseModel"""

            logger.info(u"Running experiments for {} and model {}".format(processor.name, model.__class__.__name__))

            results_handler = results.SemiSupervisedResultsHandler(
                save_path=results_save_path, labels=processor.labels, target=processor.target
            )

            experiment_params = dict(
                confidence_threshold=self._confidence_threshold,
                max_iterations=self._max_iterations,
                evaluation_size=self._evaluation_size
            )

            if self._minimum_instances is not None:
                experiment_params['minimum_instances'] = self._minimum_instances

            experiment = semisupervised.SemiSupervisedExperiment(processor, model, **experiment_params)

            experiment.run(results_handler)

            results_handler.save_results()

    def run(self):
        logger.info(u"Running semi-supervised experiments pipeline for whole corpus")

        for corpus_index, annotated_corpus in enumerate(self._corpus_iterator):
            if corpus_index < self._starting_lemma:
                logger.info(u"Skipping experiments pipeline for lemma {}. ".format(annotated_corpus.lemma) +
                            u"The corpus has already been parsed.")

            if not annotated_corpus.has_multiple_senses() or annotated_corpus.lemma == u"estar":
                logger.info(u"Skipping experiments pipeline for lemma {}. ".format(annotated_corpus.lemma) +
                            u"The corpus doesn't have enough senses")
                continue

            unannotated_corpus = self._unannotated_corpus_iterator[annotated_corpus.lemma]

            logger.info(u"Running experiments pipeline for lemma {} with index {}"
                        .format(annotated_corpus.lemma, corpus_index))

            self._run_for_corpus(annotated_corpus, unannotated_corpus, corpus_index)

            logger.info(u"Finished experiments pipeline for lemma {}".format(annotated_corpus.lemma))
