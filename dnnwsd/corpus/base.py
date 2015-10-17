# -*- coding: utf-8 -*-


class Word(object):
    def __init__(self, token, tag=None, lemma=None, is_main_verb=False):
        assert isinstance(token, unicode) and (lemma is None or isinstance(lemma, unicode))

        self.token = token
        self.tag = tag
        self.lemma = lemma
        self.is_main_verb = is_main_verb

    def __unicode__(self):
        tag = self.tag if self.tag else u""
        lemma = self.lemma if self.lemma else u""
        verb = u"verb" if self.is_main_verb else u""

        return u"{} {} {} {}".format(self.token, lemma, tag, verb).strip()

    def __str__(self):
        return unicode(self).encode("utf-8")

    def __repr__(self):
        return str(self)


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

    def __unicode__(self):
        return "\n".join(map(lambda (i, w): u"{:03d} {}".format(i, unicode(w)), enumerate(self)))

    def __str__(self):
        return unicode(self).encode("utf-8")

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._words)

    def predicate_window(self, window_size):
        start = max(0, self.predicate_index - window_size)
        end = min(len(self._words), self.predicate_index + window_size + 1)

        return self._words[start:end]

    def predicate(self):
        return self._words[self.predicate_index]


class CorpusDirectoryIterator(object):
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir

    def __iter__(self):
        raise NotImplementedError


class Corpus(object):
    def __init__(self, lemma):
        assert isinstance(lemma, unicode)

        self.lemma = lemma
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

    def tokens(self, window_size=None):
        """
        Method to return all the tokens for every predicate window of every sentence.
        Useful to get collocations.
        :param window_size: Size of the window. If None, return all tokens in the sentence.
        :return: A list of the tokens.
        """

        for sentence in self:
            window_size = len(sentence) if window_size is None else window_size

            for word in sentence.predicate_window(window_size):
                yield word.token
