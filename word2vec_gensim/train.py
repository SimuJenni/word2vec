#!/usr/bin/env python3

import logging
import os
import pickle
import sys
from pathlib import Path

import gensim
from gensim.models.word2vec import Text8Corpus

from nltk.corpus import brown, movie_reviews, treebank, reuters, gutenberg

data_dir = Path('../word2vec_data')


# Choose training corpus

corpus, corpus_name = Text8Corpus('../word2vec_data/text8'), 'text8'
corpus, corpus_name = brown.sents(), 'brown'
corpus, corpus_name = movie_reviews.sents(), 'movies'



sg = 0   # if 1, Skip-Gram else CBOW
parameters = [
    ('hs', [0, 1]),                              # if 1, hierarchical softmax else negative sampling
    ('alpha', [4**x for x in range(-5, 5)]),     # learning rate
    ('window', list(range(2, 12))),              # window size
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


def train(corpus, param, value):
    """Compute, save and return word2vec model with `param` set to `value.`

    Only load the pre-computed model if it already exists on disk.
    """
    default_params = dict(sentences=corpus, iter=10, workers=os.cpu_count(), sg=sg)
    # Note: cpu_count() also counts "logical" cores (Hyper-threading))

    fname = '{}_{}_{}'.format(corpus_name, param, value)
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
    model.save(dest_path.as_posix())
    return model


def main():
    setup()
    run_trainings()


if __name__ == '__main__':
    main()
