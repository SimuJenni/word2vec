#!/usr/bin/env python3
import pickle
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt


source_path = Path('../word2vec_data/eval_cbow')


Result = namedtuple('Result', ['corpus', 'parameter', 'value', 'skipgram', 'fpath'])


def parse_filename(fpath):
    fpath = Path(fpath)
    corpus, parameter, value = fpath.name.split('_')
    skipgram = False  # TODO
    return Result(corpus, parameter, value, skipgram, fpath)


def plot_parameter_graph(parameter, results):
    fig = plt.figure()
    # ax1 = fig.add_axes([1, 1, 1, 1])
    ax1 = fig.add_subplot(111)

    plot_values = []
    for fpath in source_path.iterdir():
        if not fpath.match('*_%s_*' % parameter):
            continue
        result = parse_filename(fpath)
        with fpath.open('rb') as f:
            data = pickle.load(f)
        accuracy = sum(
            len(sect['correct']) / (len(sect['correct']) + len(sect['incorrect']))
            for sect in data
        )
        plot_values.append((int(result.value), accuracy))

    label = "corpus: {}".format("Text8")  # TODO

    # ax1.plot(plot_values, label=label)
    xs, ys = zip(*sorted(plot_values))
    ax1.plot(xs, ys, label=label)
    # ax1.set_xticks(xticks)

    ax1.legend(loc='best')
    fig.suptitle("Accuracy")
    fig.savefig('abcd.png')

plot_parameter_graph('window', 33)
