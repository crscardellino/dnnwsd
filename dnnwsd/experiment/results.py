# -*- coding: utf-8 -*-

import os
import numpy as np

from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _format_matrix(matrix):
    return "\n".join(",".join(map(lambda e: u"{:.02f}".format(e), row)) for row in matrix)


class ResultsHandler(object):
    def __init__(self, experiment_name, save_path, labels, target=list()):
        self._experiment_name = experiment_name
        self._save_path = save_path
        self._labels = labels
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.fscores = []
        self.most_common_precision = []
        self.less_common_recall = []
        self.target_counts = [c[0] for c in Counter(target).most_common()]

    def add_result(self, y_true, y_pred):
        self.accuracies.append(accuracy_score(y_true, y_pred))

        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(len(self._labels))
        )
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.fscores.append(fscore)

        if self.target_counts:
            self.most_common_precision.append(precision[self.target_counts[0]])
            self.less_common_recall.append(recall[self.target_counts[1:]].mean())

    def save_results(self):
        with open(os.path.join(self._save_path, "accuracy"), "w") as f:
            f.write("\n".join(map(lambda a: "{:.02f}".format(a), self.accuracies)) + "\n")

        with open(os.path.join(self._save_path, "precision"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.precisions))

        with open(os.path.join(self._save_path, "recall"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.recalls))

        with open(os.path.join(self._save_path, "fscores"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.fscores))

        if self.target_counts:
            with open(os.path.join(self._save_path, "most_common_precision"), "w") as f:
                f.write("\n".join(map(lambda a: "{:.02f}".format(a), self.most_common_precision)) + "\n")

            with open(os.path.join(self._save_path, "less_common_recall"), "w") as f:
                f.write("\n".join(map(lambda a: "{:.02f}".format(a), self.less_common_recall)) + "\n")


class SemiSupervisedResultsHandler(ResultsHandler):
    def __init__(self, **kwargs):
        super(SemiSupervisedResultsHandler, self).__init__(**kwargs)

        self.test_accuracies = []
        self.test_precisions = []
        self.test_recalls = []
        self.test_fscores = []
        self.evaluation_sentences = []
        self.target_distributions = []

    def add_test_result(self, y_true, y_pred):
        self.test_accuracies.append(accuracy_score(y_true, y_pred))

        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(len(self._labels))
        )
        self.test_precisions.append(precision)
        self.test_recalls.append(recall)
        self.test_fscores.append(fscore)

    def add_evaluation_sentences(self, evaluation_sentences):
        self.evaluation_sentences.append(evaluation_sentences)

    def add_target_distribution(self, target_distribution):
        """
        :type target_distribution: collections.Counter
        """

        self.target_distributions.append(defaultdict(int, target_distribution))

    def save_results(self):
        super(SemiSupervisedResultsHandler, self).save_results()

        with open(os.path.join(self._save_path, "test_accuracy"), "w") as f:
            f.write("\n".join(map(lambda a: "{:.02f}".format(a), self.test_accuracies)) + "\n")

        with open(os.path.join(self._save_path, "test_precision"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.test_precisions))

        with open(os.path.join(self._save_path, "test_recall"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.test_recalls))

        with open(os.path.join(self._save_path, "test_fscores"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.test_fscores))

        with open(os.path.join(self._save_path, "evaluation.txt"), "w") as f:
            for iteration, sentences in enumerate(self.evaluation_sentences, start=1):
                f.write("{}\n".format("="*13))
                f.write("Iteration {:03d}\n".format(iteration))
                f.write("{}\n".format("="*13))

                for sentence, target in sentences:
                    f.write(u"{} --- {}\n".format(target, sentence).encode("utf-8"))

                f.write("\n\n")

        with open(os.path.join(self._save_path, "targets_distribution"), "w") as f:
            for iteration, target_distribution in enumerate(self.target_distributions):
                for idx, label in enumerate(self._labels):
                    f.write(u"{:03d},{},{}\n".format(iteration, label, target_distribution[idx]))
