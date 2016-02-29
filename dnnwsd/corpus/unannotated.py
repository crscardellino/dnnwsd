# -*- coding: utf-8 -*-

import logging
import os
import unicodedata

from .base import Word, Sentence, Corpus, CorpusDirectoryIterator
from ..utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _get_word(line, is_main_lemma, corpus_name):
    assert corpus_name in {'sensem', 'semeval'}

    word_info = line.replace(u'\x00', ' ').split()

    # We consider only the two first characters in case of sensem or the whole tag in case of semeval

    tag = word_info[3][:2] if corpus_name == 'sensem' else word_info[3]

    return Word(word_info[1], tag=tag, lemma=word_info[2], is_main_lemma=is_main_lemma)


def _filter_symbols(word):
    return not word.tag.startswith("F")


class UnannotatedCorpusDirectoryIterator(CorpusDirectoryIterator):
    def __init__(self, corpus_dir, corpus_name="sensem"):
        assert corpus_name in {'sensem', 'semeval'}
        super(UnannotatedCorpusDirectoryIterator, self).__init__(corpus_dir)
        self._corpus_name = corpus_name

    def __iter__(self):
        for fname in sorted((fin for fin in os.listdir(self._corpus_dir) if fin != "lemmas")):
            fpath = os.path.join(self._corpus_dir, fname)
            lemma = self.lemmas[int(fname)]

            assert isinstance(lemma, unicode)

            logger.info(u"Getting unannotated corpus from lemma {}".format(lemma).encode("utf-8"))

            yield UnannotatedCorpus(lemma, fpath, self._corpus_name)

    def __getitem__(self, item):
        fname = "{:03}".format(self.lemmas.index(item))
        fpath = os.path.join(self._corpus_dir, fname)

        logger.info(u"Getting unannotated corpus from lemma {}".format(item).encode("utf-8"))

        return UnannotatedCorpus(item, fpath)


class UnannotatedCorpus(Corpus):
    def __init__(self, lemma, fpath, corpus_name='sensem'):
        assert isinstance(lemma, unicode)
        assert corpus_name in {'sensem', 'semeval'}

        super(UnannotatedCorpus, self).__init__(lemma)

        self._corpus_name = corpus_name

        logger.info(u"Reading sentences from file {}".format(fpath).encode("utf-8"))

        with open(fpath, "r") as f:
            raw_sentences = f.read().decode("utf-8")
            raw_sentences = raw_sentences.strip().split("\n\n")

        logger.info(u"Parsing sentences from file {}".format(fpath).encode("utf-8"))

        for sentence in raw_sentences:
            sentence = unicodedata.normalize("NFC", sentence).split("\n")
            lemma_info = sentence.pop(0).split()
            lemma_position = int(lemma_info[2])

            assert len(lemma_info) == 3

            words = filter(_filter_symbols, map(
                lambda (i, l): _get_word(l, i == lemma_position, self._corpus_name), enumerate(sentence, start=1)
            ))

            predicate_index = map(lambda w: w.is_main_lemma, words).index(True)

            self._sentences.append(Sentence(words, predicate_index))

        logger.info(u"All sentences parsed in file {}".format(fpath).encode("utf-8"))
