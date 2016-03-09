# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np
import os
import shutil
import tensorflow as tf
import unicodedata

from ..experiment.ladder import LadderNetworksExperiment
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _write_results(results, evaluations, results_path):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path)
    os.makedirs(os.path.join(results_path, 'evaluations'))

    accuracy = []
    mcp = []
    lcr = []
    validation_accuracy = []
    validation_mcp = []
    validation_lcr = []

    for result in results:
        accuracy.append((result['initial_accuracy'], result['final_accuracy']))
        mcp.append((result['initial_mcp'], result['final_mcp']))
        lcr.append((result['initial_lcr'], result['final_lcr']))
        validation_accuracy.append(result['validation_accuracy'])
        validation_mcp.append(result['validation_mcp'])
        validation_lcr.append(result['validation_lcr'])

    accuracy = np.array(accuracy, dtype=np.float32).mean(axis=0)  # Initial and final accuracy mean
    mcp = np.array(mcp, dtype=np.float32).mean(axis=0)  # Initial and final mcp mean
    lcr = np.array(lcr, dtype=np.float32).mean(axis=0)  # Initial and final lcr mean
    validation_accuracy = np.array(validation_accuracy, dtype=np.float32).mean(axis=0) # Initial and final validation_accuracy mean
    validation_mcp = np.array(validation_mcp, dtype=np.float32).mean(axis=0)  # Initial and final validation_mcp mean
    validation_lcr = np.array(validation_lcr, dtype=np.float32).mean(axis=0)  # Initial and final validation_lcr mean

    np.savetxt(os.path.join(results, 'accuracy'), accuracy, fmt='%.2f')
    np.savetxt(os.path.join(results, 'mcp'), mcp, fmt='%.2f')
    np.savetxt(os.path.join(results, 'lcr'), lcr, fmt='%.2f')
    np.savetxt(os.path.join(results, 'validation_accuracy'), validation_accuracy, fmt='%.2f')
    np.savetxt(os.path.join(results, 'validation_mcp'), validation_mcp, fmt='%.2f')
    np.savetxt(os.path.join(results, 'validation_lcr'), validation_lcr, fmt='%.2f')

    for eidx, evaluation in enumerate(evaluations):
        epath = os.path.join(results_path, 'evaluations', '{:02d}.txt'.format(eidx))
        with open(epath, 'w') as f:
            for epoch, sentences in evaluation:
                f.write("{}\n".format("="*13))
                f.write("Iteration {:03d}\n".format(epoch))
                f.write("{}\n".format("="*13))

                for sentence in sentences:
                    f.write(sentence.encode("utf-8"))

                f.write("\n\n")


class LadderNetworksPipeline(object):
    _experiments = {
        'bow': 'Bag of Words',
        'vec': 'Word Vectors',
        'vecpos': 'Word Vectors with PoS'
    }

    def __init__(self, dataset_directory, dataset_indexes, lemmas_path,
                 results_directory, layers, denoising_cost, **kwargs):
        self._dataset_directory = dataset_directory
        self._dataset_indexes = dataset_indexes
        self._results_directory = results_directory

        with open(lemmas_path, 'r') as f:
            self._lemmas = unicodedata.normalize("NFC", f.read().decode("utf-8")).strip().split("\n")

        self._layers = layers
        self._denoising_cost = denoising_cost
        self._repetitions = kwargs.pop('repetitions', 5)
        self._epochs = kwargs.pop('epochs', 10)
        self._noise_std = kwargs.pop('noise_std', 0.3)
        self._starter_learning_rate = kwargs.pop('starter_learning_rate', 0.02)
        self._train_ratio = kwargs.pop('train_ratio', 0.8)
        self._test_ratio = kwargs.pop('test_ratio', 0.1)
        self._validation_ratio = kwargs.pop('validation_ratio', 0.1)

    def run(self):
        for dataset_index in self._dataset_indexes:
            logger.info(u"Running set of experiments for lemma {}".format(self._lemmas[dataset_index]))

            for experiment, experiment_name in self._experiments.iteritems():
                logger.info(u"Running {} experiments".format(experiment_name))

                dataset_path = os.path.join(self._dataset_directory, experiment, "{:03d}.p".format(dataset_index))
                results_path = os.path.join(self._results_directory, experiment)

                results = []
                evaluations = []

                for repetition in xrange(self._repetitions):
                    logger.info(u"Running repetition {}".format(repetition + 1))

                    with tf.Graph().as_default() as g:
                        layers = copy.copy(self._layers[experiment])
                        denoising_cost = copy.copy(self._denoising_cost[experiment])

                        ladder_experiment = LadderNetworksExperiment(
                            dataset_path, layers, denoising_cost,
                            epochs=self._epochs, noise_std=self._noise_std,
                            starter_learning_rate=self._starter_learning_rate, train_ratio=self._train_ratio,
                            test_ratio=self._test_ratio, validation_ratio=self._validation_ratio
                        )

                        ladder_experiment.run()

                        logger.info(u"Finished experiments for repetition {} - {} experiment - lemma {}".format(
                            repetition+1, experiment_name, self._lemmas[dataset_index]
                        ))

                        results.append(
                            copy.deepcopy(ladder_experiment.results)
                        )

                        evaluations.append([])  # repetition evaluations
                        for (eidx, epoch) in enumerate(ladder_experiment.evaluation_sentences):
                            evaluations[repetition].append([])  # epoch evaluations
                            for (eval_sent, y_pred) in epoch:
                                raw_sentence = ladder_experiment.dataset[eval_sent]
                                sense = ladder_experiment.dataset.labels[y_pred]

                                evaluations[repetition][eidx].append(
                                    u"{} -- {}".format(sense, raw_sentence)
                                )

                logger.info(u"Finished all the {} experiment repetitions".format(experiment_name))

                _write_results(results, evaluations, results_path)

            logger.info(u"Finished all the experiments for lemma {}".format(self._lemmas[dataset_index]))
