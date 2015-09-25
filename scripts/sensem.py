#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from bs4 import BeautifulSoup
from lxml import etree

ifile = sys.argv[1]
ofile = sys.argv[2]

tree = etree.parse(ifile)

with open(ofile, 'w') as outf:
    for sentence in tree.findall('.//content'):
        outf.write(re.sub(r'\s\s+', ' ', BeautifulSoup(
            etree.tostring(sentence, encoding='UTF-8'), 'lxml'
        ).get_text().replace('\n', ' ').strip()).encode('utf-8'))
