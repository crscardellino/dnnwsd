# -*- coding: utf-8 -*-

import nltk
import os

from lxml import etree
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

lemmatizer = WordNetLemmatizer()

filename = "../resources/semeval/train/english-lexical-sample.train.xml"
output_dir = "../resources/semeval/lexelts"

lemmas = []
tree = etree.parse(filename)

for lexelt_idx, lexelt in enumerate(tree.findall("lexelt")):
    lexelt_lemma = lexelt.attrib['item']
    lemmas.append(lexelt_lemma)

    print "Analysing {} (number {})".format(lexelt_lemma, lexelt_idx)

    sentences = []

    for idx, instance in enumerate(lexelt.findall("instance"), start=1):
        senseid = instance.find("answer").attrib['senseid']
        senseid = "{}-{:02d}".format(lexelt_lemma, int(senseid))

        tokens = etree.tostring(instance.find("context")).split()[1:-1]
        lexelt_index = tokens.index("<head>")
        tokens.remove("<head>")
        tokens.remove("</head>")
        lexelt_token = tokens[lexelt_index]

        tagged_tokens = nltk.pos_tag(tokens)

        assert tagged_tokens[lexelt_index][0] == lexelt_token

        sentence = ["#{:05d} {} {}".format(idx, senseid, lexelt_index+1)]

        for idx, (token, tag) in enumerate(tagged_tokens):
            wordnet_pos = get_wordnet_pos(tag)

            if wordnet_pos != '':
                token_lemma = lemmatizer.lemmatize(token, wordnet_pos)
            else:
                token_lemma = token.lower()

            if idx != lexelt_index:
                sentence.append("{:03d} {} {} {}".format(idx+1, token, token_lemma, tag))
            else:
                sentence.append("{:03d} {} {} {} lemma".format(idx+1, token, token_lemma, tag))

        sentences.append("\n".join(sentence))

    with open(os.path.join(output_dir, "{:03d}".format(lexelt_idx)), 'w') as f:
        for sentence in sentences:
            f.write("{}\n\n".format(sentence))


with open(os.path.join("lemmas"), "w") as f:
    f.write("\n".join(lemmas))
