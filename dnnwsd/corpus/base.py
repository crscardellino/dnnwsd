# -*- coding: utf-8 -*-


class Word(object):
    def __init__(self, token, idx=None, tag=None, lemma=None):
        assert isinstance(token, unicode) and (lemma is None or isinstance(lemma, unicode))

        self.token = token
        self.idx = idx
        self.tag = tag
        self.lemma = lemma


class Sentence(object):
    def __init__(self, words, predicate_index, sense=u'?'):
        assert sense is None or isinstance(sense, unicode)

        self._words = words
        self.predicate_index = predicate_index
        self.sense = sense

    def __iter__(self):
        for word in self._words:
            yield word

    def __getitem__(self, item):
        return self._words[item]

    def predicate_window(self, window_size):
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
        """:type : list of Sentence"""

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence
