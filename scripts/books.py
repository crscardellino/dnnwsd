#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from bs4 import BeautifulSoup
from lxml import etree

ifile = sys.argv[1]
ofile = sys.argv[2]

tree = etree.parse(ifile)

with open(ofile, 'w') as outf:
    outf.write(BeautifulSoup(
        etree.tostring(tree.find('body')), 'lxml'
    ).get_text())
