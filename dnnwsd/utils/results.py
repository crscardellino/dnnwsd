# -*- coding: utf-8 -*-

from sklearn import metrics


class ResultsHandler(object):
    def __init__(self):
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.fscores = []

    def add_result(self, y_true, y_pred):
        self.accuracies.append(metrics.accuracy_score(y_true, y_pred))
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_true, y_pred)

        self.precisions.append(precision)
        self.recalls.append(recall)
        self.fscores.append(fscore)
