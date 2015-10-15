# -*- coding: utf-8 -*-

import logging.config
import yaml
from os import path


def setup_logging():
    file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.yaml')
    with open(file_path, 'rt') as f:
        config = yaml.load(f.read())
    logging.config.dictConfig(config)
