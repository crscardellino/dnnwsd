# -*- coding: utf-8 -*-

import os
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _format_matrix(matrix):
    return "\n".join(",".join(map(lambda e: u"{:.02f}".format(e), row)) for row in matrix)


class ResultsHandler(object):
    def __init__(self, experiment_name, save_path, labels):
        self._experiment_name = experiment_name
        self._save_path = save_path
        self._labels = labels
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.fscores = []

    def add_result(self, y_true, y_pred):
        self.accuracies.append(accuracy_score(y_true, y_pred))

        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(len(self._labels))
        )
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.fscores.append(fscore)

    def save_results(self):
        with open(os.path.join(self._save_path, "accuracy"), "w") as f:
            f.write("\n".join(map(lambda a: u"{:.02f}".format(a), self.accuracies)) + "\n")

        with open(os.path.join(self._save_path, "precision"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.precisions))

        with open(os.path.join(self._save_path, "recall"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.recalls))

        with open(os.path.join(self._save_path, "fscores"), "w") as f:
            f.write(",".join(self._labels).encode("utf-8") + "\n")
            f.write(_format_matrix(self.fscores))
