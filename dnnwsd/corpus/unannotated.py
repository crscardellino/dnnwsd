# -*- coding: utf-8 -*-

import logging
import os
import unicodedata

from .base import Word, Sentence, Corpus, CorpusDirectoryIterator
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _get_word(line, is_main_verb):
    word_info = line.split()

    return Word(word_info[1], tag=word_info[3][:2], lemma=word_info[2], is_main_verb=is_main_verb)


def _filter_symbols(word):
    return not word.tag.startswith("F")


class UnannotatedCorpusDirectoryIterator(CorpusDirectoryIterator):
    def __init__(self, corpus_dir, filtered_verbs):
        super(UnannotatedCorpusDirectoryIterator, self).__init__(corpus_dir)
        self._filtered_verbs = set(self.verbs.index(verb) for verb in filtered_verbs)

    def __iter__(self):
        for fname in (fin for fin in os.listdir(self._corpus_dir)
                      if fin != "verbs" and int(fin) in self._filtered_verbs):
            fpath = os.path.join(self._corpus_dir, fname)
            lemma = self.verbs[int(fname)]

            assert isinstance(lemma, unicode)

            logger.info(u"Getting corpus from lemma {}".format(lemma).encode("utf-8"))

            yield UnannotatedCorpus(lemma, fpath)


class UnannotatedCorpus(Corpus):
    def __init__(self, lemma, fpath):
        assert isinstance(lemma, unicode)

        super(UnannotatedCorpus, self).__init__(lemma)

        logger.info(u"Reading sentences from file {}".format(fpath).encode("utf-8"))

        with open(fpath, "r") as f:
            raw_sentences = f.read().decode("utf-8")
            raw_sentences = raw_sentences.strip().split("\n\n")

        logger.info(u"Parsing sentences from file {}".format(fpath).encode("utf-8"))

        for sentence in raw_sentences:
            sentence = unicodedata.normalize("NFC", sentence).split("\n")
            lemma_info = sentence.pop(0).split()
            verb_position = int(lemma_info[2])

            assert len(lemma_info) == 3

            words = filter(_filter_symbols, map(
                lambda (i, l): _get_word(l, i == verb_position), enumerate(sentence, start=1)
            ))

            predicate_index = map(lambda w: w.is_main_verb, words).index(True)

            self._sentences.append(Sentence(words, predicate_index))

        logger.info(u"All sentences parsed in file {}".format(fpath).encode("utf-8"))