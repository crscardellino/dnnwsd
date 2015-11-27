#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gensim
import os
import shutil
import sys
import yaml

from dnnwsd.pipeline import semisupervised


def run_pipeline(corpus_dir, unannotated_corpus_dir, results_dir, experiments, features_path, **kwargs):
    pipeline = semisupervised.SemiSupervisedPipeline(
        corpus_dir, unannotated_corpus_dir, results_dir, experiments, features_path, **kwargs
    )

    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script to run experiments")
    parser.add_argument("config_file", help="YAML Configuration File")

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f.read())

    results_directory = os.path.join(config['results']['directory'])

    if os.path.isdir(results_directory):
        overwrite = raw_input("Want to overwrite results? (Y/n)").strip().lower()
        overwrite = "y" if overwrite == "" else overwrite

        if overwrite.startswith("y"):
            shutil.rmtree(results_directory)
        else:
            sys.exit(1)

    os.makedirs(results_directory)

    shutil.copy2(args.config_file, results_directory)

    print >> sys.stderr, "Loading word2vec model"
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format(
        config['pipeline']['word2vec_model_path'], binary=True
    )

    corpus_directory = config['annotated_corpus']['directory']
    unannotated_corpus_directory = config['unannotated_corpus']['directory']
    features_path = config['unannotated_corpus']['features_path']

    experiment_set = []

    for experiment in config['pipeline']['experiments']:
        pkey = experiment['processor']
        pparam = {
            'window_size': config['pipeline']['processor_config']['window_size']
        }

        if pkey == 'bow':
            pparam['vocabulary_filter'] = config['pipeline']['processor_config']['vocabulary_filter']
        elif pkey in {'wordvec', 'wordvecpos'}:
            pparam['word2vec_model'] = word2vec_model

            if pkey == 'wordvecpos':
                pparam['pos_tags_path'] = os.path.join(features_path, 'pos_tags')

        mkey = experiment['model']
        mparam = config['pipeline']['model_config'] if mkey == 'mlp' else {}

        experiment_set.append((pkey, pparam, mkey, mparam))

    configuration = dict(
        max_iterations=config['pipeline']['max_iterations'],
        confidence_threshold=config['pipeline']['confidence_threshold'],
        evaluation_size=config['pipeline']['evaluation_size'],
        sense_filter=config['annotated_corpus']['sense_filter']
    )

    run_pipeline(corpus_directory, unannotated_corpus_directory, results_directory,
                 experiment_set, features_path, **configuration)

    print >> sys.stderr, "Finished all experiments"
