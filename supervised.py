#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gensim
import itertools
import os
import shutil
import sys
import yaml

from dnnwsd.pipeline import supervised


def run_pipeline(experiments, corpus_dir, results_dir, configuration):
    pipeline = supervised.SupervisedPipeline(
        corpus_dir, results_dir, iterations=configuration['pipeline']['iterations'], experiment_set=experiments
    )

    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script to run experiments")
    parser.add_argument("config_file", help="YAML Configuration File")

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f.read())

    results_directory = os.path.join(config['results']['directory'],
                                     "sense_filter_{}".format(config['corpus']['sense_filter'])).decode("utf-8")
    os.makedirs(results_directory)

    shutil.copy2(args.config_file, results_directory)

    corpus_directory = config['corpus']['directory'].decode("utf-8")
    iterations = config['pipeline']['iterations']

    print >> sys.stderr, "Loading word2vec model"
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format(
        config['pipeline']['word2vec_model_path'], binary=True
    )

    for widx, window_size in enumerate(config['pipeline']['window_sizes']):
        print >> sys.stderr, "Running experiment for window of size {}".format(window_size)

        rdir = os.path.join(results_directory, "window_{}".format(window_size))
        os.makedirs(rdir)

        experiment_set = []

        for pkey, mkey in itertools.product(config['pipeline']['processors'], config['pipeline']['models']):
            pparam = {'window_size': window_size}
            mparam = {}

            if pkey == 'wordvec':
                pparam['word2vec_model'] = word2vec_model
            else:
                pparam['vocabulary_filter'] = config['pipeline']['vocabulary_filter']

            if pkey in {'bopos', 'pos'}:
                pparam['pos_filter'] = config['pipeline']['pos_filter']

            if mkey == 'autoencoder':
                mparam = {
                    'fine_tune_epochs': config['pipeline']['fine_tune_epochs'],
                    'pre_train_epochs': config['pipeline']['pre_train_epochs'],
                    'batch_size': config['pipeline']['batch_size'],
                    'activation': config['pipeline']['activation']
                }

                if pkey != 'wordvec':
                    mparam['layer'] = config['pipeline']['encoder_layer']
                else:
                    mparam['layer'] = config['pipeline']['encoder_wordvec_layer'][widx]

            experiment_set.append((pkey, pparam, mkey, mparam))

        run_pipeline(experiment_set, corpus_directory, rdir, config)

    print >> sys.stderr, "Finished all experiments"
