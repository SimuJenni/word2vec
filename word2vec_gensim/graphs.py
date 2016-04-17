#!/usr/bin/env python3
import pickle
import sys
from collections import namedtuple
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as pyplot


source_path = Path('../word2vec_data/')
pretty = {
    'window': 'window size',
    'size': 'vector size',
    'alpha': 'learning rate (alpha)',
    'hs': 'hierarchical softmax'
}


class Result(namedtuple('Result', ['corpus', 'parameter', 'value', 'skipgram', 'fpath'])):
    """Represent the resulting file of an accuracy evaluation"""

    def load(self):
        with self.fpath.open('rb') as f:
            return pickle.load(f)

    @classmethod
    def parse_filename(cls, fpath):
        fpath = Path(fpath)
        corpus, parameter, value = fpath.name.split('_')
        skipgram = fpath.resolve().match('eval_sg/*')
        return cls(corpus, parameter, value, skipgram, fpath)


def plot_parameter_graph(parameter):
    """Plot the accuracy of word2vec models for different values of 'parameter'

    Plot for each corpus, both for skipgram and cbow"""
    fig = pyplot.figure()
    plt = fig.add_subplot(111)

    xticks = None
    for (corpus, skipgram), results in find_parameter_cases(parameter):
        results = list(sorted(results, key=lambda r: float(r.value)))

        if not xticks:
            xticks = ['{:.3g}'.format(float(r.value)) for r in results]
            print(xticks)

        plot_values = []
        for result in results:
            data = result.load()
            accuracies = [
                len(sect['correct']) / (len(sect['correct']) + len(sect['incorrect']))
                for sect in data if len(sect['correct']) + len(sect['incorrect']) > 0
            ]
            average = sum(accuracies) / len(accuracies)
            plot_values.append(average)

        skipgram_str = 'skipgram' if skipgram else 'cbow'
        label = "corpus: {}, {}".format(corpus, skipgram_str)
        plt.plot(range(len(results)), plot_values, label=label)

    plt.set_xticks(range(len(results)))
    plt.set_xticklabels(xticks)
    plt.set_xlabel(pretty[parameter].capitalize())

    plt.set_ylabel('Accuracy')
    plt.legend(loc='best', prop={'size': 10})
    fig.suptitle("Accuracy for different values of {}".format(pretty[parameter]))

    graph_path = source_path / 'graphs'
    graph_path.mkdir(exist_ok=True)
    graph_path /= 'graph_acc_{}.pdf'.format(parameter)
    fig.savefig(graph_path.as_posix())


def find_parameter_cases(parameter):
    """Return results grouped by their value of skipgram and corpus.

    Output example: [
        (('Text8', False), [<results>]),
        (('Text8', True), [<results>]),
        ...
    ]"""
    pattern = 'eval_*/*_%s_*' % parameter
    results = [Result.parse_filename(f) for f in source_path.glob(pattern)]

    def groupkey(result):
        return (result.corpus, result.skipgram)

    results.sort(key=groupkey)
    return groupby(results, key=groupkey)


def main():
    if len(sys.argv) < 2:
        print("Usage: graphs.py <parameter>")
        sys.exit(1)

    plot_parameter_graph(sys.argv[1])


if __name__ == '__main__':
    main()
