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

    corpus_directory = config['corpus']['directory']
    iterations = config['pipeline']['iterations']

    print >> sys.stderr, "Loading word2vec model"
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format(
        config['pipeline']['word2vec_model_path'], binary=True
    )

    experiment_set = []

    for processor, model in itertools.product(config['pipeline']['processors'], config['pipeline']['models']):
        model = model.copy()  # To modify it without touching the original

        pkey = processor
        pparam = {'window_size': config['pipeline']['processors_defaults']['window_size']}

        if pkey in {'bow', 'bopos', 'pos'}:
            pparam['vocabulary_filter'] = config['pipeline']['processors_defaults']['vocabulary_filter']

        if pkey in {'bopos', 'pos'}:
            pparam['pos_filter'] = config['pipeline']['processors_defaults']['pos_filter']

        if pkey == 'wordvec':
            pparam['word2vec_model'] = word2vec_model

        mkey = model.pop('type')
        mparam = model

        mparam.update(config['pipeline']['models_defaults'].get(mkey, {}))

        experiment_set.append((pkey, pparam, mkey, mparam))

    run_pipeline(experiment_set, corpus_directory, results_directory, config)

    print >> sys.stderr, "Finished all experiments"
