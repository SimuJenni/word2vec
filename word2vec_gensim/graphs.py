#!/usr/bin/env python3
import pickle
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt


Result = namedtuple('Result', ['corpus', 'parameter', 'value', 'skipgram', 'fpath'])


def parse_filename(fpath):
    fpath = Path(fpath)
    corpus, parameter, value = fpath.name.split('_')
    skipgram = False  # TODO
    return Result(corpus, parameter, value, skipgram, fpath)


def plot_parameter_graph(parameter, results):
    for fpath
