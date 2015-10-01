#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

ifile = sys.argv[1]
ofile = sys.argv[2]

paragraph_flag = False

with open(ifile, 'r') as inf:
    with open(ofile, 'w') as outf:
        for line in inf.readlines():
            if line.strip().startswith("<doc ") or line.strip() == "</doc>" or line.strip() == "":
                if paragraph_flag:
                    outf.write('\n')
                    paragraph_flag = False
                continue

            paragraph_flag = True

            line = re.sub(r'a\s*\.\s*m\s*\.', 'am', line.decode('utf-8').strip(), flags=re.UNICODE).strip()
            line = re.sub(r'p\s*\.\s*m\s*\.', 'pm', line, flags=re.UNICODE).strip()
            sentences = re.sub(r'\.\s+|\.$', '\n', line, flags=re.UNICODE).strip().split('\n')

            for idx, sentence in enumerate(sentences, start=1):
                outl = re.sub(r'[\W_]+', ' ', sentence, flags=re.UNICODE).strip()
                outl = re.sub(r'\d', 'DIGITO ', outl, flags=re.UNICODE).strip()
                outl = re.sub(r'\s+', ' ', outl, flags=re.UNICODE)  # Remove double spaces

                outl_size = len(outl.split())

                outf.write(outl.encode('utf-8'))

                if idx != len(sentences):
                    outf.write('\n')
                elif line.endswith('.'):
                    outf.write('\n')
                    paragraph_flag = False
                else:
                    outf.write(' ')
