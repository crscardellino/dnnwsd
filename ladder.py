#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from dnnwsd.pipeline import ladder

LAYERS = {
    'bow': [2000, 1000, 500],
    'vec': [5000, 3000, 1000],
    'vecpos': [5000, 3000, 1000]
}
DENOISING_COST = {
    'bow': [1000.0, 10.0, 0.1, 0.1, 0.1],
    'vec': [1000.0, 10.0, 0.1, 0.1, 0.1],
    'vecpos': [1000.0, 10.0, 0.1, 0.1, 0.1]
}
DATA_INDEXES = [2, 14, 26, 90, 139, 140]
DATASET_DIRECTORY = "/home/ccardellino/datasets/dataset_corpus/es/7k/"
LEMMAS_PATH = "/home/ccardellino/datasets/dataset_corpus/es/7k/lemmas"
RESULTS_DIRECTORY = "results/"

pipeline = ladder.LadderNetworksPipeline(
    DATASET_DIRECTORY,
    DATA_INDEXES,
    LEMMAS_PATH,
    RESULTS_DIRECTORY,
    LAYERS,
    DENOISING_COST,
    repetitions=10,
    epochs=10,
    noise_std=0.3,
    starter_learning_rate=0.01,
    train_ratio=0.8,
    test_ratio=0.1,
    validation_ratio=0.1,
    evaluation_amount=10,
    population_growth_count=1000
)

print >> sys.stderr, "Starting set of experiments"

pipeline.run()

print >> sys.stderr, "All experiments are finished"
