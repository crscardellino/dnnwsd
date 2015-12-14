# -*- coding: utf-8 -*-

import logging
import os
import re
import unicodedata

from collections import defaultdict

from .base import Word, Sentence, Corpus, CorpusDirectoryIterator
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _get_word(line):
    word_info = line.split()
    is_main_verb = len(word_info) == 5 and word_info[4] == u'lemma'

    return Word(word_info[1], tag=word_info[3], lemma=word_info[2], is_main_verb=is_main_verb)


def _filter_symbols(line):
    word_info = line.split()

    return len(word_info) > 3 and \
        word_info[3][0] in {"C", "D", "E", "F", "I", "J", "L", "M", "N", "P", "R", "T", "U", "V", "W"}


class SemevalCorpusDirectoryIterator(CorpusDirectoryIterator):
    def __init__(self, corpus_dir, sense_filter=3):
        super(SemevalCorpusDirectoryIterator, self).__init__(corpus_dir)
        self._sense_filter = sense_filter

    def __get_verbs__(self):
        with open(os.path.join(self._corpus_dir, "lemmas"), "r") as f:
            self.verbs = f.read().decode("utf-8").strip().split("\n")

    def __iter__(self):
        for fname in sorted((fin for fin in os.listdir(self._corpus_dir) if fin != "lemmas")):
            fpath = os.path.join(self._corpus_dir, fname)
            lemma = self.verbs[int(fname)]

            logger.info(u"Getting corpus from lemma {}".format(lemma).encode("utf-8"))

            yield SemevalCorpus(lemma, fpath, self._sense_filter)


class SemevalCorpus(Corpus):
    def __init__(self, lemma, fpath, sense_filter=3):
        assert isinstance(lemma, unicode)

        super(SemevalCorpus, self).__init__(lemma)

        self.senses = defaultdict(int)

        logger.info(u"Reading sentences from file {}".format(fpath).encode("utf-8"))

        with open(fpath, "r") as f:
            raw_sentences = f.read().decode("utf-8")
            raw_sentences = re.sub(r"\n\n\n+", "\n\n", raw_sentences.strip(), flags=re.UNICODE).split("\n\n")

        logger.info(u"Parsing sentences from file {}".format(fpath).encode("utf-8"))

        for sentence in raw_sentences:
            sentence = unicodedata.normalize("NFC", sentence).split("\n")
            sense_info = sentence.pop(0).split()

            if len(sense_info) != 3:
                logger.info(u"Ignoring sentence {} of lemma {} and sense {}"
                            .format(sense_info[0], self.lemma, sense_info[1]).encode("utf-8"))
                continue

            words = map(_get_word, filter(_filter_symbols, sentence))
            try:
                predicate_index = map(lambda w: w.is_main_verb, words).index(True)
            except ValueError:
                logger.info(u"Ignoring sentence {} of lemma {} and sense {}"
                            .format(sense_info[0], self.lemma, sense_info[1]).encode("utf-8"))
                continue

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
