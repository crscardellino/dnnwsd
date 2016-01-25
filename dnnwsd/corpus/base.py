# -*- coding: utf-8 -*-

import os
import re
import unicodedata


_tokens_with_symbols = {
    u"ee.uu.": u"EEUU",
    u"estados_unidos": u"EEUU",
    u"u.s.a.": u"USA",
    u"sr.": u"sr",
    u"sra.": u"sra",
    u"srta.": u"srta",
    u"a.m.": u"am",
    u"p.m.": u"pm"
}


class Word(object):
    def __init__(self, token, tag=None, lemma=None, is_main_lemma=False):
        assert isinstance(token, unicode) and (lemma is None or isinstance(lemma, unicode))

        self.token = ""

        if token.lower() in _tokens_with_symbols:
            self.token = _tokens_with_symbols[token.lower()]
        elif re.match(r"^([\-])?\d+([.,]\d+)?$", token):
            self.token = u"DIGITO"
        else:
            self.token = token

        self.tag = tag.upper()
        self.lemma = lemma
        self.is_main_lemma = is_main_lemma

    def __unicode__(self):
        tag = self.tag if self.tag else u""
        lemma = self.lemma if self.lemma else u""
        main_lemma = u"lemma" if self.is_main_lemma else u""

        return u"{} {} {} {}".format(self.token, lemma, tag, main_lemma).strip()

    def __str__(self):
        return unicode(self).encode("utf-8")

    def __repr__(self):
        return str(self)

    def tokens_and_lemmas(self):
        """
        Method to get different combinations of tokens and lemmas, with different use of capital letters
        in order to look for it in a embedding model.
        :return: A list of all possible combinations of a token or a lemma, ordered by importance
        """

        return [
            self.token,
            self.token.lower(),
            self.token.capitalize(),
            self.token.upper(),
            self.lemma,
            self.lemma.capitalize(),
            self.lemma.upper()
        ]


class Sentence(object):
    def __init__(self, words, predicate_index, sense=u'?'):
        assert isinstance(sense, unicode)

        self._words = words
        """:type : list of dnnwsd.corpus.base.Word"""
        self.predicate_index = predicate_index
        """:type : int"""
        self.sense = sense
        """:type : unicode"""

    def __iter__(self):
        for word in self._words:
            yield word

    def __getitem__(self, item):
        return self._words[item]

    def __unicode__(self):
        return "\n".join(map(lambda (i, w): u"{:03d} {}".format(i, unicode(w)), enumerate(self)))

    def __str__(self):
        return unicode(self).encode("utf-8")

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._words)

    def predicate_window(self, window_size):
        """
        Gives the window around the predicate. If the window size is zero or less returns all the words.
        :param window_size: Size of the window
        :return: Words in the window.
        """
        window_size = len(self) if window_size <= 0 else window_size

        start = max(0, self.predicate_index - window_size)
        end = min(len(self), self.predicate_index + window_size + 1)

        return self._words[start:end]

    def predicate(self):
        return self._words[self.predicate_index]


class CorpusDirectoryIterator(object):
    def __init__(self, corpus_dir):
        self._corpus_dir = corpus_dir
        self.lemmas = []
        self.__get_lemmas__()

    def __get_lemmas__(self):
        with open(os.path.join(self._corpus_dir, "lemmas"), "r") as f:
            self.lemmas = unicodedata.normalize("NFC", f.read().decode("utf-8")).strip().split("\n")

    def __iter__(self):
        raise NotImplementedError


class Corpus(object):
    def __init__(self, lemma):
        assert isinstance(lemma, unicode)

        self.lemma = unicodedata.normalize("NFC", lemma)
        self._sentences = []
        """:type : list of dnnwsd.corpus.base.Sentence"""

    def __iter__(self):
        for sentence in self._sentences:
            yield sentence

    def __getitem__(self, item):
        return self._sentences[item]

    def __unicode__(self):
        return "\n\n".join(map(lambda s: unicode(s), self._sentences))

    def __str__(self):
        return unicode(self).encode("utf-8")

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._sentences)

    def slice_indexes(self, indexes):
        for index in sorted(indexes, reverse=True):
            del self._sentences[index]

    def tokens(self, window_size=0):
        """
        Method to return all the tokens for every predicate window of every sentence.
        Useful to get collocations.
        :param window_size: Size of the window. If 0, return all tokens in the sentence.
        :return: A list of the tokens.
        """

        for sentence in self:
            for word in sentence.predicate_window(window_size):
                yield word.token

    def pos_tags(self, window_size=0):
        for sentence in self:
            for word in sentence.predicate_window(window_size):
                yield word.tag
