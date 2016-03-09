#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Add dnnwsd to the path
import sys
from os import path

sys.path.append(path.abspath(path.dirname('../')))

import cPickle as pickle
import uuid
import numpy as np

import gensim

# Change the log config file to relative path
from dnnwsd.utils import setup_logging
setup_logging.CONFIG_FILE = u"../config/logging.yaml"

from dnnwsd.corpus import semeval, unannotated
from dnnwsd.processor import vecprocessor

annotated_corpus_directory = "../resources/semeval/lexelts"
unannotated_corpus_directory = "../../wikicorpus/en/wikicorpus_lemmas_sample_7k/"
pos_tags_file = "../resources/semisupervised_features/en/pos_tags"
corpus_datasets_dir = "../resources/corpus_datasets/en/7k/vecpos"

annotated_corpus_directory_iterator = semeval.SemevalCorpusDirectoryIterator(annotated_corpus_directory)
unannotated_corpus_directory_iterator = unannotated.UnannotatedCorpusDirectoryIterator(unannotated_corpus_directory,
                                                                                       corpus_name='sensem')

word_vectors_path = "../resources/wordvectors/GoogleNews-vectors-negative300.bin.gz"

word2vec_model = gensim.models.Word2Vec.load_word2vec_format(word_vectors_path, binary=True)

for corpus_index, annotated_corpus in enumerate(annotated_corpus_directory_iterator):
    if not annotated_corpus.has_multiple_senses():
        print u"Skipping preprocess for corpus of lemma {}".format(annotated_corpus.lemma)
        continue

    unannotated_corpus = unannotated_corpus_directory_iterator[annotated_corpus.lemma]

    vec_processor = vecprocessor.SemiSupervisedWordVectorsPoSProcessor(
        annotated_corpus, unannotated_corpus, word2vec_model, pos_tags_file)

    vec_processor.instances()

    annotated_dataset = dict(data=vec_processor.dataset, target=vec_processor.target, labels=vec_processor.labels)
    sentences_ids = []
    unannotated_sentences = {}

    for sentence in unannotated_corpus:
        sentence_id = str(uuid.uuid4())
        raw_sentence = []

        for word in sentence:
            word_token = u"_{}_".format(word.token) if word.is_main_lemma else word.token
            raw_sentence.append(word_token)

        raw_sentence = " ".join(raw_sentence)

        sentences_ids.append(sentence_id)
        unannotated_sentences[sentence_id] = raw_sentence

    unannotated_dataset = dict(data=vec_processor.unannotated_dataset, sentences=np.array(sentences_ids))

    lemma_dataset = dict(
        lemma=annotated_corpus.lemma,
        index="{:03d}".format(corpus_index),
        annotated_dataset=annotated_dataset,
        unannotated_dataset=unannotated_dataset,
        unannotated_sentences=unannotated_sentences
    )

    corpus_dataset = path.join(corpus_datasets_dir, "{:03d}.p".format(corpus_index))

    with open(corpus_dataset, "wb") as f:
        pickle.dump(lemma_dataset, f)
