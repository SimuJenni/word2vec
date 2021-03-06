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

valid_corpora = ['wiki', 'text8', 'brown']


class Result(namedtuple('Result', ['corpus', 'parameter', 'value', 'skipgram', 'fpath'])):
    """Represent the resulting file of an accuracy evaluation"""

    def load(self):
        try:
            with self.fpath.open('rb') as f:
                return pickle.load(f)
        except EOFError:
            return None

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
        if corpus not in valid_corpora:
            continue

        if not xticks:
            if parameter == 'alpha':
                xticks = ['${2^{%s}}$' % r.value for r in results]
            else:
                xticks = ['{:.3g}'.format(float(r.value)) for r in results]
            print(xticks)

        plot_values = []
        for result in results:
            data = result.load()
            if data is not None:
                accuracies = [
                    len(sect['correct']) / (len(sect['correct']) + len(sect['incorrect']))
                    for sect in data if len(sect['correct']) + len(sect['incorrect']) > 0
                ]
                average = sum(accuracies) / len(accuracies)
                plot_values.append(average)

        skipgram_str = 'skipgram' if skipgram else 'cbow'
        label = "corpus: {}, {}".format(corpus, skipgram_str)
        plt.plot(range(len(plot_values)), plot_values, label=label)

    plt.set_xticks(range(len(results)))
    plt.set_xlim([0, len(results) - 1])  # avoid blank space
    plt.set_xticklabels(xticks)
    plt.set_xlabel(pretty[parameter].capitalize())
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.set_ylim([0, 1])

    plt.set_ylabel('Accuracy')
    plt.legend(loc='best', prop={'size': 10})
    plt.grid()
    fig.suptitle("Accuracy for different values of {}".format(pretty[parameter]))

    if parameter == 'alpha':
        # Change text renderer to be able to render 2**x
        pyplot.rc('text', usetex="True")
        pyplot.rcParams['font.size'] = 13
        # pyplot.rcParams['font.sans-serif'] = 'Bitstream Vera Sans'
        # pyplot.rcParams['font.size'] = 14
        # pyplot.rcParams['font.weight'] = 'bold'
        plt.tick_params(axis='x', which='major', labelsize=16)

    graph_path = source_path / 'graphs'
    graph_path.mkdir(exist_ok=True)
    graph_path /= 'graph_acc_{}.png'.format(parameter)
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

    # Temporarily ignore brands
    results = [r for r in results if not r.fpath.resolve().match('*/eval_*_brands/*')]

    def groupkey(result):
        return (result.corpus, result.skipgram)

    results.sort(key=groupkey)
    return groupby(results, key=groupkey)


def plot_accuracy_graph():
    """
        Plot the k-NN accuracy of word2vec models
    """
    pattern = 'accuracy/*_*_*'
    results = [f for f in source_path.glob(pattern)]

    fig = pyplot.figure()
    plt = fig.add_subplot(111)

    for result in results:
        fpath = Path(result)
        model_type, corpus, parameter, value = fpath.name.split('_')
        try:
            with fpath.open('rb') as f:
                data = pickle.load(f)
        except EOFError:
            return None
        label = "{} {}".format(model_type.upper(), corpus)
        accuracy = data[len(data)-1]['accuracy']
        plt.plot(range(len(accuracy)), accuracy, label=label)

    plt.set_ylabel('Accuracy')
    plt.set_xlabel('k')
    plt.legend(loc='best', prop={'size': 10})
    plt.grid()
    fig.suptitle("k-NN Accuracy ")

    graph_path = source_path / 'graphs'
    graph_path.mkdir(exist_ok=True)
    graph_path /= 'graph_knn-acc.png'
    fig.savefig(graph_path.as_posix())

def main():
    if len(sys.argv) < 2:
        print("Usage: graphs.py <parameter>")
        sys.exit(1)

    #plot_parameter_graph(sys.argv[1])
    plot_accuracy_graph()


if __name__ == '__main__':
    main()
