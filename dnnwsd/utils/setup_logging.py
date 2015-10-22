# -*- coding: utf-8 -*-

import logging.config
import yaml

from os import path

CONFIG_FILE = u"config/logging.yaml"


def setup_logging():
    file_path = path.join(CONFIG_FILE)
    with open(file_path, 'rt') as f:
        config = yaml.load(f.read())
    logging.config.dictConfig(config)
