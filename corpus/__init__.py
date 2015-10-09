# -*- coding: utf-8 -*-


class Word(object):
    def __init__(self, token, idx=None, tag=None, lemma=None, sense=None):
        self.token = token
        self.idx = idx
        self.tag = tag
        self.lemma = lemma
        self.sense = sense


class Sentence(object):
    def __init__(self, words, pidx):
        self._words = words
        self._pidx = pidx

    def __iter__(self):
        for word in self._words:
            yield word

    def __getitem__(self, item):
        return self._words[item]

    def predicate(self):
        return self._words[self._pidx]

    def tokens(self):
        for word in self:
            yield word.token
