#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import subprocess
import sys
from bs4 import BeautifulSoup
from collections import defaultdict
from lxml import etree

FREELING_COMMAND = "analyze -f es.cfg --noloc --noner --nonumb --nodate --noquant".split()

corpus_file = sys.argv[1]
output_dir = sys.argv[2]

print >> sys.stderr, "Loading Corpus"
corpus = etree.parse(corpus_file)

print >> sys.stderr, "Parsing Corpus"

sentences = corpus.findall(".//sentence")
total = len(sentences)

verbs = defaultdict(int)
senses = defaultdict(int)

for idx, sentence in enumerate(sentences, start=1):
    print >> sys.stderr, "Parsing sentence {} of {}".format(idx, total)

    sentence_id = sentence.attrib["id"]
    lexical = sentence.find("lexical")

    verb = lexical.attrib["verb"]
    verbs[verb] += 1
    verb_id = "{:05d}".format(verbs[verb])

    sense = u"{}-{}".format(verb, lexical.attrib["sense"])
    senses[sense] += 1

    verb_forms = {vw.text for vw in sentence.findall(".//word[@verb='true']")}

    raw_sentence = re.sub(r'\s\s+', ' ',
                          BeautifulSoup(
                              etree.tostring(sentence.find("content"), encoding='UTF-8'), 'lxml'
                          ).get_text().replace('\n', ' ').strip())

    proc = subprocess.Popen(FREELING_COMMAND, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    (pipe_out, pipe_err) = proc.communicate(input=raw_sentence.encode('UTF-8'))

    if pipe_err.strip() != '':
        print >> sys.stderr, pipe_err
        sys.exit(1)

    words = [w.decode('UTF-8').strip().split()[:3] for w in pipe_out.split('\n') if w.strip() != '']

    verb_found = False

    for widx, word in enumerate(words):
        if word[1] == verb and word[0] in verb_forms:
            if not verb_found:
                word.append("verb")
                verb_found = True
            else:
                word.append("verb")
                print >> sys.stderr, \
                    u"Possible conflict in file {} - sentence {} - sense {} - corpus sentence {}".format(
                        verb, verb_id, sense, sentence_id).encode('UTF-8')

    if not verb_found:
        print >> sys.stderr, u"Verb not found in file {} - sentence {} - sense {} - corpus sentence".format(
            verb, verb_id, sense, sentence_id).encode('UTF-8')

    with open(os.path.join(output_dir, verb), "a") as fout:
        fout.write("#{} ".format(verb_id) + sense.encode('UTF-8') + '\n')

        for word in words:
            fout.write('\t'.join(word).encode('UTF-8') + '\n')

        fout.write('\n')

print >> sys.stderr, "\nAll corpus parsed"
print >> sys.stderr, "Saving stats of senses"

with open("senses_stats.txt", "w") as fout:
    for sense in sorted(senses):
        fout.write(u"{} {}\n".format(sense, senses[sense]).encode('UTF-8'))
