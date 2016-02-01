# -*- coding: utf-8 -*-

import cPickle as pickle
import os
from dnnwsd.corpus import semeval, unannotated
from collections import defaultdict

corpus_iter = semeval.SemevalCorpusDirectoryIterator("resources/semeval/lexelts")
unannotated_corpus_iter = unannotated.UnannotatedCorpusDirectoryIterator("../wikicorpus/en/wikicorpus_lemmas")

tags = set()

for annotated_corpus in corpus_iter:
    if not annotated_corpus.has_multiple_senses():
        print u"Skipping experiments pipeline for lemma {}.".format(annotated_corpus.lemma)
        print u"The corpus doesn't have enough senses"
        continue

    tokens = defaultdict(int)
    unannotated_corpus = unannotated_corpus_iter[annotated_corpus.lemma]

    for sentence in annotated_corpus:
        for word in sentence.predicate_window(5):
            tags.add(word.tag)
            tokens[word.token] += 1

    for sentence in unannotated_corpus:
        for word in sentence.predicate_window(5):
            tags.add(word.tag)
            tokens[word.token] += 1

    fname = "{:03}.p".format(corpus_iter.verbs.index(annotated_corpus.lemma))
    fpath = os.path.join("../wikicorpus/en/bow_features", fname)
    with open(fpath, "wb") as f:
        pickle.dump(tokens, f)

with open("../wikicorpus/en/bow_features/pos_tags", "w") as f:
    f.write("\n".join(sorted(tags)).encode("utf-8"))
