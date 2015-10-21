# -*- coding: utf-8 -*-

import cPickle as pickle
import logging
import os
import re

from collections import defaultdict

from .base import Word, Sentence, Corpus, CorpusDirectoryIterator
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _get_word(line):
    word_info = line.split()
    is_main_verb = len(word_info) == 5 and word_info[4] == u'verb'

    return Word(word_info[1], tag=word_info[3][:2], lemma=word_info[2], is_main_verb=is_main_verb)


def _filter_symbols(line):
    word_info = line.split()

    return len(word_info) > 3 and not word_info[3].startswith("F")


class SenSemCorpusDirectoryIterator(CorpusDirectoryIterator):
    def __iter__(self):
        for fname in os.listdir(self._corpus_dir):
            fpath = os.path.join(self._corpus_dir, fname).decode("utf-8")
            lemma = fname.decode("utf-8")

            logger.info(u"Getting corpus from lemma {}".format(lemma).encode("utf-8"))

            if not self._is_binary:
                yield SenSemCorpus(lemma, fpath)
            else:
                yield pickle.load(fpath)


class SenSemCorpus(Corpus):
    def __init__(self, lemma, fpath, sense_filter=3):
        assert isinstance(fpath, unicode)

        super(SenSemCorpus, self).__init__(lemma)

        self.senses = defaultdict(int)

        logger.info(u"Reading sentences from file {}".format(fpath).encode("utf-8"))

        with open(fpath, "r") as f:
            raw_sentences = f.read().decode("utf-8")
            raw_sentences = re.sub(r"\n\n\n+", "\n\n", raw_sentences.strip(), flags=re.UNICODE).split("\n\n")

        logger.info(u"Parsing sentences from file {}".format(fpath).encode("utf-8"))

        for sentence in raw_sentences:
            sentence = sentence.split("\n")
            sense_info = sentence.pop(0).split()

            if len(sense_info) != 3:
                logger.info(u"Ignoring sentence {} of lemma {} and sense {}"
                            .format(sense_info[0], self.lemma, sense_info[1]).encode("utf-8"))
                continue

            words = map(_get_word, filter(_filter_symbols, sentence))
            predicate_index = map(lambda w: w.is_main_verb, words).index(True)

            self._sentences.append(Sentence(words, predicate_index, sense_info[1]))
            self.senses[sense_info[1]] += 1

        if sense_filter > 1:
            logger.info(u"Filtering senses with less than {} instances in file {}".format(
                sense_filter, fpath).encode("utf-8")
            )

            self.senses = {sense: count for sense, count in self.senses.iteritems() if count >= sense_filter}
            self._sentences = filter(lambda s: s.sense in self.senses, self._sentences)

        logger.info(u"All sentences parsed in file {}".format(fpath).encode("utf-8"))

    def has_multiple_senses(self):
        return len(self.senses) > 1

    def get_senses(self):
        return ((sense, count) for sense, count in self.senses.iteritems())
