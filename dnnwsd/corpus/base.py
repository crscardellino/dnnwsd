# -*- coding: utf-8 -*-


class Word(object):
    def __init__(self, token, idx=None, tag=None, lemma=None):
        self.token = token
        self.idx = idx
        self.tag = tag
        self.lemma = lemma


class Sentence(object):
    def __init__(self, words, predicate_index, sense):
        self._words = words
        self.predicate_index = predicate_index
        self.sense = sense

    def __iter__(self):
        for word in self._words:
            yield word

    def __getitem__(self, item):
        return self._words[item]

    def predicate_window(self, window_size=5):
        start = max(0, self.predicate_index - window_size)
        end = min(len(self._words), self.predicate_index + window_size + 1)

        return self._words[start:end]

    def predicate(self):
        return self._words[self.predicate_index]

    def tokens(self):
        for word in self:
            yield word.token


class CorpusDirectoryIterator(object):
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir

    def __iter__(self):
        raise NotImplementedError


class Corpus(object):
    def __init__(self, lemma):
        assert isinstance(lemma, unicode)

        self.lemma = lemma
        self.sentences = []

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence
