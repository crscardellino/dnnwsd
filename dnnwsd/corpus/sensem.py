# -*- coding: utf-8 -*-

import logging
import os
import re
from collections import defaultdict
from .tokens import Word, Sentence
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def get_word_from_line(line):
    word_info = line.split()

    return Word(word_info[1], idx=int(word_info[0])-1, tag=word_info[3][:2], lemma=word_info[2])


class SenSemCorpus(object):
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir

    def __iter__(self):
        for fname in os.listdir(self.corpus_dir):
            fpath = os.path.join(self.corpus_dir, fname).decode("utf-8")
            lemma = fname.decode("utf-8")

            logger.info(u"Getting corpus from lemma {}".format(lemma).encode("utf-8"))

            yield SenSemLemmaCorpus(lemma, fpath)


class SenSemLemmaCorpus(object):
    def __init__(self, lemma, fpath):
        assert isinstance(lemma, unicode) and isinstance(fpath, unicode)

        self.lemma = lemma
        self.sentences = []
        self.senses = defaultdict(int)

        logger.info(u"Reading sentences from file {}".format(fpath).encode("utf-8"))

        with open(fpath, "r") as f:
            raw_sentences = f.read().decode("utf-8")
            raw_sentences = re.sub(r"\n\n\n+", "\n\n", raw_sentences, flags=re.UNICODE).split("\n\n")

        logger.info(u"Parsing sentences from file {}".format(fpath).encode("utf-8"))

        for sentence in raw_sentences:
            sentence = sentence.split("\n")
            sense_info = sentence.pop(0).split()

            if len(sense_info) != 3:
                logger.info(u"Ignoring sentence {} of lemma {} and sense {}"
                            .format(sense_info[0], self.lemma, sense_info[1]).encode("utf-8"))
                continue

            words = map(get_word_from_line, filter(lambda l: len(l.split()) == 4, sentence))

            self.sentences.append(Sentence(words, int(sense_info[2])-1, sense_info[1]))
            self.senses[sense_info[1]] += 1

        logger.info(u"All sentences parsed in file {}".format(fpath).encode("utf-8"))

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence

    def has_multiple_senses(self):
        return len(self.senses) > 1

    def get_filtered_senses(self, sense_filter=3):
        return ((sense, count) for sense, count in self.senses.iteritems() if count >= sense_filter)

    def get_filtered_sentences(self, sense_filter=3):
        return (sentence for sentence in self if self.senses[sentence.sense] >= sense_filter)
