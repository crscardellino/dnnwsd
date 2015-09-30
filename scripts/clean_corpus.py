#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

ifile = sys.argv[1]
ofile = sys.argv[2]

with open(ifile, 'r') as inf:
    with open(ofile, 'w') as outf:
        for line in inf.readlines():
            if line.strip().startswith("<doc ") or line.strip() == "</doc>" or line.strip() == "":
                continue

            sentences = re.sub(r'\.\s+|\.$', '\n', line.decode('utf-8').strip(), flags=re.UNICODE).strip().split('\n')

            for idx, sentence in enumerate(sentences, start=1):
                outl = re.sub(r'[\W_]+', ' ', sentence, flags=re.UNICODE).strip()
                outl = outl.replace('0', 'cero ').strip()
                outl = outl.replace('1', 'uno ').strip()
                outl = outl.replace('2', 'dos ').strip()
                outl = outl.replace('3', 'tres ').strip()
                outl = outl.replace('4', 'cuatro ').strip()
                outl = outl.replace('5', 'cinco ').strip()
                outl = outl.replace('6', 'seis ').strip()
                outl = outl.replace('7', 'siete ').strip()
                outl = outl.replace('8', 'ocho ').strip()
                outl = outl.replace('9', 'nueve ').strip()
                outl = re.sub(r'\s+', ' ', outl, flags=re.UNICODE)  # Remove double spaces

                outf.write(outl.encode('utf-8'))
                if idx != len(sentences):
                    outf.write('\n')
                else:
                    outf.write(' ')
