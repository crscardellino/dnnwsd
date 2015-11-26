# -*- coding: utf-8 -*-

import logging
import numpy as np
import scipy.sparse as sp

from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from .base import Experiment, TRAIN_RATIO
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SemiSupervisedExperiment(Experiment):
    def __init__(self, processor, model, **kwargs):
        super(SemiSupervisedExperiment, self).__init__(processor, model)
        self._processor = processor
        """:type : dnnwsd.processor.vecprocessor.SemiSupervisedWordVectorsProcessor"""
        self._confidence_threshold = kwargs.pop("confidence_threshold", 0.99)
        self._minimum_instances = kwargs.pop("minimum_instances", int(self._processor.dataset.shape[0] * 0.01) + 1)
        self._max_iterations = kwargs.pop("max_iterations", 100)
        self._evaluation_size = kwargs.pop("evaluation_size", 10)
        self._max_accuracy = 0.0

    def split_dataset(self):
        tr_set = set()
        te_set = set()
        va_set = set()

        init_tr_index = []
        init_te_index = []
        init_va_index = []

        permuted_indices = np.random.permutation(self._processor.target.shape[0])

        # We make sure every split has at least one example of each class
        for target_index in permuted_indices:
            if self._processor.target[target_index] not in tr_set:
                init_tr_index.append(target_index)
                tr_set.add(self._processor.target[target_index])
            elif self._processor.target[target_index] not in te_set:
                init_te_index.append(target_index)
                te_set.add(self._processor.target[target_index])
            elif self._processor.target[target_index] not in va_set:
                init_va_index.append(target_index)
                va_set.add(self._processor.target[target_index])

        filtered_indices = permuted_indices[~np.in1d(
            permuted_indices, np.array(init_tr_index + init_te_index + init_va_index)
        )]

        # We randomly split the remaining examples
        tr_index, te_index = train_test_split(filtered_indices, train_size=TRAIN_RATIO)
        split_index = int(te_index.shape[0] / 2) + 1
        te_index, va_index = te_index[:split_index], te_index[split_index:]

        return (np.hstack([init_tr_index, tr_index]),
                np.hstack([init_te_index, te_index]),
                np.hstack([init_va_index, va_index])
                )

    def _evaluate_sentences(self, candidates, target_candidates):
        possible_targets = {t: None for t in target_candidates}

        for idx, target in np.random.permutation(list(enumerate(target_candidates))):
            if not possible_targets[target]:
                possible_targets[target] = idx

        evaluation_sentences = []

        for target, index in possible_targets.iteritems():
            ex = candidates[index]
            sentence = []

            for word in self._processor.unannotated_corpus[ex]:
                word_token = u"_{}_".format(word.token) if word.is_main_verb else word.token
                sentence.append(word_token)

            sentence = " ".join(sentence)

            evaluation_sentences.append((sentence, target))

        possible_target_values = set(possible_targets.values())

        evaluation_candidates =\
            [(idx, ex) for idx, ex in enumerate(candidates) if idx not in possible_target_values]

        evaluation_size = max(0, self._evaluation_size - len(evaluation_sentences))

        for idx, ex in np.random.permutation(evaluation_candidates)[:evaluation_size]:
            sentence = []

            for word in self._processor.unannotated_corpus[ex]:
                word_token = u"_{}_".format(word.token) if word.is_main_verb else word.token
                sentence.append(word_token)

            sentence = " ".join(sentence)

            target_sense = self._processor.labels[target_candidates[idx]]

            evaluation_sentences.append((sentence, target_sense))

        return evaluation_sentences

    def _run_bootstrap(self, results_handler, supervised_dataset):
        """
        :type results_handler: dnnwsd.experiment.results.SemiSupervisedResultsHandler
        """
        logger.info(u"Getting initial validation results")

        results_handler.add_result(supervised_dataset['y_val'], self._model.predict(supervised_dataset['X_val']))

        results_handler.add_target_distribution(supervised_dataset['y_train'])

        self._max_accuracy = results_handler.accuracies[-1]

        logger.info(u"Initial validation accuracy: {:.02f}".format(self._max_accuracy))

        for iteration in xrange(1, self._max_iterations + 1):
            logger.info(u"Running iteration {} of {}".format(iteration, self._max_iterations))

            logger.info(u"Getting candidates to automatically tag")
            probabilities = self._model.predict_proba(self._processor.unannotated_dataset)
            candidates = np.where(probabilities.max(axis=1) >= self._confidence_threshold)[0]

            dataset_candidates = self._processor.unannotated_dataset[candidates]
            target_candidates = probabilities[candidates].argmax(axis=1)

            logger.info(u"Fitting dataset with automatically annotated candidates")

            if supervised_dataset['X_train'] == np.ndarray:
                stacked_data = np.vstack(
                    (supervised_dataset['X_train'], self._processor.automatic_dataset, dataset_candidates)
                )
            else:  # If is not an array, then is a sparse matrix
                stacked_data = sp.vstack(
                    (supervised_dataset['X_train'], self._processor.automatic_dataset, dataset_candidates)
                )

            self._model.fit(
                stacked_data,
                np.hstack((supervised_dataset['y_train'], self._processor.automatic_target, target_candidates))
            )

            new_accuracy = accuracy_score(
                supervised_dataset['y_val'],
                self._model.predict(supervised_dataset['X_val'])
            )
            logger.info(u"New validation accuracy: {:.02f}".format(new_accuracy))

            if new_accuracy <= self._max_accuracy - 0.1:
                logger.info(
                    u"Truncating at iteration {} for a large drop in the accuracy - Max: {:.02f} - Current {:.02f}"
                    .format(iteration, self._max_accuracy, new_accuracy)
                )
                break

            if candidates.shape[0] < self._minimum_instances:
                logger.info(
                    u"Truncating at iteration {}. Only {} instances were selected (the minimum being {})."
                    .format(iteration, candidates.shape[0], self._minimum_instances)
                )
                break

            self._max_accuracy = max(self._max_accuracy, new_accuracy)

            self._processor.tag_slice(candidates, target_candidates)

            results_handler.add_result(
                supervised_dataset['y_val'],
                self._model.predict(supervised_dataset['X_val'])
            )

            results_handler.add_evaluation_sentences(self._evaluate_sentences(candidates, target_candidates))

            target_distribution = Counter(
                np.hstack((supervised_dataset['y_train'], self._processor.automatic_target))
            )

            results_handler.add_target_distribution(target_distribution)

            if self._processor.untagged_corpus_proportion()[0] == 0:
                logger.info(
                    u"Truncating at iteration {}. No more instances to add."
                    .format(iteration, candidates.shape[0], self._minimum_instances)
                )
                break

        logger.info(u"Final validation accuracy: {:.02f}".format(results_handler.accuracies[-1]))

    def run(self, results_handler):
        """
        :type results_handler: dnnwsd.experiment.results.SemiSupervisedResultsHandler
        """
        logger.info(u"Splitting the dataset")

        tr_index, te_index, va_index = self.split_dataset()

        supervised_dataset = dict(
            X_train=self._processor.dataset[tr_index],
            y_train=self._processor.target[tr_index],
            X_test=self._processor.dataset[te_index],
            y_test=self._processor.target[te_index],
            X_val=self._processor.dataset[va_index],
            y_val=self._processor.target[va_index]
        )

        logger.info(u"Fitting the supervised dataset in the classifier")

        self._model.fit(supervised_dataset['X_train'], supervised_dataset['y_train'])

        logger.info(u"Getting test results from the supervised dataset")

        results_handler.add_test_result(
            supervised_dataset['y_test'],
            self._model.predict(supervised_dataset['X_test'])
        )

        logger.info(u"Initial test accuracy: {:.02f}".format(results_handler.test_accuracies[0]))

        logger.info(u"Starting bootstrap iterations")

        self._run_bootstrap(results_handler, supervised_dataset)

        logger.info(u"Finished bootstrap iterations. Saving final test results.")

        logger.info(u"Fitting final model")

        if supervised_dataset['X_train'] == np.ndarray:
            stacked_data = np.vstack(
                (supervised_dataset['X_train'], self._processor.automatic_dataset)
            )
        else:  # If is not an array, then is a sparse matrix
            stacked_data = sp.vstack(
                (supervised_dataset['X_train'], self._processor.automatic_dataset)
            )

        self._model.fit(
            stacked_data,
            np.hstack((supervised_dataset['y_train'], self._processor.automatic_target))
        )

        results_handler.add_test_result(
            supervised_dataset['y_test'],
            self._model.predict(supervised_dataset['X_test'])
        )

        logger.info(u"Final test accuracy: {:.02f}".format(results_handler.test_accuracies[1]))
