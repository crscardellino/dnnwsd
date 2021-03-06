#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class CorpusIterator(object):
    def __init__(self, directory):
        self.directory = directory

    def __iter__(self):
        for fname in os.listdir(self.directory):
            for line in open(os.path.join(self.directory, fname)):
                if len(line.split()) < 3:  # Probably is a title or plain noise
                    continue
                else:
                    yield line.split()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word2Vec algorithm")
    parser.add_argument("train", type=str, metavar="CORPUS_DIRECTORY", help="Path to the corpus directory.")
    parser.add_argument("--output", type=str, metavar="OUTPUT", help="File to dump the word vectors (in both formats).",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors"))
    parser.add_argument("--vocab", type=str, metavar="VOCAB_OUTPUT", help="File to dump the vocabulary",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary"))
    parser.add_argument("--size", type=int, metavar="SIZE", help="Set size of word vectors.",
                        default=300)
    parser.add_argument("--window", type=int, metavar="WINDOW", help="Max skip length between words.",
                        default=10)
    parser.add_argument("--threads", type=int, metavar="THREADS", help="Set the number of threads for parallelizing.",
                        default=12)
    parser.add_argument("--min_count", type=int, metavar="MIN_COUNT",
                        help="Set the minimum number of occurrences for a word", default=5)
    parser.add_argument("--sample", type=float, metavar="SAMPLE",
                        help="Threshold for configuring which higher-frequency words are randomly downsampled",
                        default=1E-6)
    parser.add_argument("--alpha", type=float, metavar="ALPHA", help="Set the starting learning rate", default=0.025)
    parser.add_argument("--iter", type=int, metavar="ITERATIONS", help="number of iterations (epochs) over the corpus.",
                        default=5)
    parser.add_argument("--negs", type=int, metavar="NEGATIVE_SAMPLING_COUNT",
                        help="If > 0, negative sampling will be used, the int for negative specifies how many " +
                             "\"noise words\" should be drawn (usually between 5-20).", default=20)

    args = parser.parse_args()

    sentences = CorpusIterator(args.train)

    model_config = {
        "size": args.size,
        "window": args.window,
        "workers": args.threads,
        "min_count": args.min_count,
        "sample": args.sample,
        "alpha": args.alpha,
        "iter": args.iter,
        "negative": args.negs,
        "sg": 1,
        "hs": 0,
    }

    print >> sys.stderr, "Creating and Training Word2Vec model on %s corpus." % sentences.directory
    model = Word2Vec(sentences, **model_config)

    print >> sys.stderr, "Saving the model in gensim format."
    model.save(args.output + ".gsbin")

    print >> sys.stderr, "Saving the model in Word2Vec format."
    model.save_word2vec_format(args.output + ".bin", fvocab=args.vocab, binary=True)
