#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import os
import shutil
import tensorflow as tf

from os import path
from dnnwsd.experiment import neuralnetwork
from dnnwsd.utils import dataset

# Dataset Configurations
DATA_INDEX = int(sys.argv[1])
FORMAT = sys.argv[2]
EXPERIMENT = sys.argv[3]
DATASET_DIRECTORY = "/home/ccardellino/datasets/dataset_corpus/es/7k/"
RESULTS_DIRECTORY = "results/"

# General Configurations
REPETITIONS = 10
EPOCHS = 100
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.1
STARTER_LEARNING_RATE = 0.01

# Multilayer Perceptron Configurations
LAYERS = [5000, 3000, 1000]
NOISE_STD = 0.3

# CNN Configurations
WINDOW_SIZE = 11
WORD_VECTOR_SIZE = 300
FILTER_SIZES = [2, 3, 4]
NUM_FILTERS = 128
L2_REG_LAMBDA = 0.01

dataset_path = path.join(DATASET_DIRECTORY, FORMAT, "{:03d}.p".format(DATA_INDEX))
dataset_instance = dataset.DataSets(
    dataset_path=dataset_path,
    train_ratio=TRAIN_RATIO,
    test_ratio=TEST_RATIO,
    validation_ratio=VALIDATION_RATIO
)

train = np.zeros((REPETITIONS, EPOCHS + 2), dtype=np.float32)
test = np.zeros((REPETITIONS, 2), dtype=np.float32)
validation = np.zeros((REPETITIONS, EPOCHS + 2), dtype=np.float32)
train_mcp = np.zeros((REPETITIONS, EPOCHS + 2), dtype=np.float32)
test_mcp = np.zeros((REPETITIONS, 2), dtype=np.float32)
validation_mcp = np.zeros((REPETITIONS, EPOCHS + 2), dtype=np.float32)
train_lcr = np.zeros((REPETITIONS, EPOCHS + 2), dtype=np.float32)
test_lcr = np.zeros((REPETITIONS, 2), dtype=np.float32)
validation_lcr = np.zeros((REPETITIONS, EPOCHS + 2), dtype=np.float32)
train_error = np.zeros((REPETITIONS, EPOCHS + 2), dtype=np.float32)

rpath = path.join(RESULTS_DIRECTORY, FORMAT, EXPERIMENT)

for rep in xrange(REPETITIONS):
    results_path = path.join(rpath, "repetition{}".format(rep), "{:03d}".format(DATA_INDEX))

    with tf.Graph().as_default() as g:
        if EXPERIMENT == 'cnn':
            experiment = neuralnetwork.ConvolutionalNeuralNetwork(
                dataset_path_or_instance=dataset_instance,
                epochs=EPOCHS,
                starter_learning_rate=STARTER_LEARNING_RATE,
                window_size=WINDOW_SIZE,
                word_vector_size=WORD_VECTOR_SIZE,
                filter_sizes=FILTER_SIZES,
                num_filters=NUM_FILTERS,
                l2_reg_lambda=L2_REG_LAMBDA,
                shift_data=False if FORMAT == "vec" else True
            )
        elif EXPERIMENT == 'mlp':
            experiment = neuralnetwork.MultilayerPerceptron(
                dataset_path_or_instance=dataset_instance,
                layers=LAYERS,
                epochs=EPOCHS,
                starter_learning_rate=STARTER_LEARNING_RATE,
                noise_std=NOISE_STD
            )
        else:
            raise Exception("Not a valid experiment")

        experiment.run(results_path)

    del experiment
    del g

    train[rep, :] = np.loadtxt(path.join(results_path, "train"), dtype=np.float32)
    test[rep, :] = np.loadtxt(path.join(results_path, "test"), dtype=np.float32)
    validation[rep, :] = np.loadtxt(path.join(results_path, "validation"), dtype=np.float32)
    train_mcp[rep, :] = np.loadtxt(path.join(results_path, "train_mcp"), dtype=np.float32)
    test_mcp[rep, :] = np.loadtxt(path.join(results_path, "test_mcp"), dtype=np.float32)
    validation_mcp[rep, :] = np.loadtxt(path.join(results_path, "validation_mcp"), dtype=np.float32)
    train_lcr[rep, :] = np.loadtxt(path.join(results_path, "train_lcr"), dtype=np.float32)
    test_lcr[rep, :] = np.loadtxt(path.join(results_path, "test_lcr"), dtype=np.float32)
    validation_lcr[rep, :] = np.loadtxt(path.join(results_path, "validation_lcr"), dtype=np.float32)
    train_error[rep, :] = np.loadtxt(path.join(results_path, "train_error"), dtype=np.float32)

rpath_mean = path.join(rpath, "mean", "{:03d}".format(DATA_INDEX))
if path.exists(rpath_mean):
    shutil.rmtree(rpath_mean)

os.makedirs(rpath_mean)

np.savetxt(os.path.join(rpath_mean, "train"), train.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "test"), test.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "validation"), validation.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "train_mcp"), train_mcp.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "test_mcp"), test_mcp.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "validation_mcp"), validation_mcp.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "train_lcr"), train_lcr.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "test_lcr"), test_lcr.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "validation_lcr"), validation_lcr.mean(axis=0), fmt="%.2f")
np.savetxt(os.path.join(rpath_mean, "train_lcr"), train_lcr.mean(axis=0), fmt="%.2f")

