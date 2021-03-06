#!/usr/bin/env python3

import logging
import os
import pickle
import sys
from pathlib import Path

import gensim
from gensim.models.word2vec import Text8Corpus, LineSentence

from nltk.corpus import brown, movie_reviews, treebank, reuters, gutenberg

data_dir = Path('../word2vec_data')

# Choose training corpus

corpus, corpus_name = Text8Corpus('../word2vec_data/text8'), 'text8'
corpus, corpus_name = brown.sents(), 'brown'
#corpus, corpus_name = movie_reviews.sents(), 'movies'
#corpus, corpus_name = treebank.sents(), 'treebank'
#corpus, corpus_name = LineSentence('../word2vec_data/text'), 'wiki'


sg = 0   # if 1, Skip-Gram else CBOW
parameters = [
    ('hs', [0, 1]),                              # if 1, hierarchical softmax else negative sampling
    # ('alpha', [2**x for x in range(-8, 1)]),     # learning rate
    ('window', list(range(2, 13))),              # window size
    ('size', list(range(25, 301, 25)))           # vector size
]


def setup():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    if not data_dir.exists():
        logging.fatal('No "../word2vec_data" folder!')
        logging.fatal('Please create it or run the script from the right directory.')
        sys.exit(1)
    for folder in ('results_cbow', 'results_sg', 'eval_sg', 'eval_cbow'):
        (data_dir / folder).mkdir(exist_ok=True)


def run_trainings():
    for param, values in parameters:
        for value in values:
            train(corpus, param, value)


def train_alphas():
    for i in range(-8, -20, 5):
        power = i / 10
        train(corpus, 'alpha', 2**power, str(power))


def train(corpus, param, value, pretty_value=None):
    """Compute, save and return word2vec model with `param` set to `value.`

    Only load the pre-computed model if it already exists on disk.
    """
    if pretty_value is None:
        pretty_value = str(value)

    default_params = dict(sentences=corpus, iter=10, workers=os.cpu_count(), sg=sg)
    # Note: cpu_count() also counts "logical" cores (Hyper-threading))

    fname = '{}_{}_{}'.format(corpus_name, param, pretty_value)
    folder = 'results_cbow' if sg == 0 else 'results_sg'
    dest_path = data_dir / folder / fname
    if dest_path.exists():
        logging.info("'{}' already exists, don't train again".format(fname))
        with dest_path.open('rb') as f:
            return pickle.load(f)

    logging.info("Start training '{}'".format(fname))
    params = default_params.copy()
    params[param] = value

    model = gensim.models.Word2Vec(**params)
    # model.init_sims(replace=True)
    model.save(dest_path.as_posix())
    return model


def main():
    setup()
    global sg

    sg = 0
    # train_alphas()
    run_trainings()

    sg = 1
    # train_alphas()
    run_trainings()

if __name__ == '__main__':
    main()
