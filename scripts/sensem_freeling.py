#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import subprocess
import sys
from bs4 import BeautifulSoup
from lxml import etree

FREELING_COMMAND = "analyze -f es.cfg --noloc --noner --nonumb --nodate --noquant".split()

corpus_file = sys.argv[1]
output_dir = sys.argv[2]

print >> sys.stderr, "Loading Corpus"
corpus = etree.parse(corpus_file)

print >> sys.stderr, "Parsing Corpus"

sentences = corpus.findall(".//sentence")
total = len(sentences)

for idx, sentence in enumerate(sentences, start=1):
    sys.stderr.write("\rParsing sentence {} of {}".format(idx, total))

    verb_words = sentence.findall(".//word[@verb='true']")

    if len(verb_words) == 0:
        continue

    lexical = sentence.find("lexical")
    verb = lexical.attrib["verb"]
    sense = "{}-{}".format(verb, lexical.attrib["sense"])

    verb_positions = [(int(vw.attrib["id"]) - 1) for vw in verb_words]

    raw_sentence = re.sub(r'\s\s+', ' ',
                          BeautifulSoup(
                              etree.tostring(sentence, encoding='UTF-8'), 'lxml'
                          ).get_text().replace('\n', ' ').strip())

    proc = subprocess.Popen(FREELING_COMMAND, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    (pipe_out, pipe_err) = proc.communicate(input=raw_sentence)

    if pipe_err.strip() != '':
        print >> sys.stderr, pipe_err
        sys.exit(1)

    words = [w.strip().split()[:3] for w in pipe_out.split('\n') if w.strip() != '']

    for idx in verb_positions:
        words[idx].append("verb")

    with open(os.path.join(output_dir, verb), "a") as fout:
        fout.write(sense.encode('UTF-8') + '\n')

        for word in words:
            fout.write('\t'.join(word).encode('UTF-8') + '\n')

        fout.write('\n')

print >> sys.stderr, "\nAll corpus parsed"
