#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Add dnnwsd to the path
import sys
from os import path

sys.path.append(path.abspath(path.dirname('../')))

import cPickle as pickle
import uuid
import numpy as np

# Change the log config file to relative path
from dnnwsd.utils import setup_logging
setup_logging.CONFIG_FILE = u"../config/logging.yaml"

from dnnwsd.corpus import sensem, unannotated
from dnnwsd.processor import bowprocessor

annotated_corpus_directory = "../resources/sensem/"
unannotated_corpus_directory = "../../wikicorpus/es/wikicorpus_lemmas_sample_30k/"
corpus_datasets = "../resources/corpus_datasets/bow_datasets_es.p"

annotated_corpus_directory_iterator = sensem.SenSemCorpusDirectoryIterator(annotated_corpus_directory)
unannotated_corpus_directory_iterator = unannotated.UnannotatedCorpusDirectoryIterator(unannotated_corpus_directory)

semisupervised_features_directory = "../resources/semisupervised_features/es/"

bow_datasets = {}

for corpus_index, annotated_corpus in enumerate(annotated_corpus_directory_iterator):
    if not annotated_corpus.has_multiple_senses() or annotated_corpus.lemma == u'estar':
        print u"Skipping preprocess for corpus of lemma {}".format(annotated_corpus.lemma)
        continue

    unannotated_corpus = unannotated_corpus_directory_iterator[annotated_corpus.lemma]
    
    semisupervised_features_path = path.join(semisupervised_features_directory, "{:03d}.p".format(corpus_index))
    
    # Bag of words datasets
    
    bow_processor = bowprocessor.SemiSupervisedBoWProcessor(annotated_corpus, unannotated_corpus,
                                                           semisupervised_features_path)
    bow_processor.instances()
    
    annotated_dataset = dict(data=bow_processor.dataset, target=bow_processor.target)
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

    unannotated_dataset = dict(data=bow_processor.unannotated_dataset, sentences=np.array(sentences_ids))
    
    bow_datasets["{:03d}".format(corpus_index)] = dict(
        annotated_dataset=annotated_dataset,
        unannotated_dataset=unannotated_dataset,
        unannotated_sentences=unannotated_sentences
    )

with open(corpus_datasets, "wb") as f:
    pickle.dump(bow_datasets, f)
