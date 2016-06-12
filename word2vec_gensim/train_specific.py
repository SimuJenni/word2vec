#!/usr/bin/env python3

import pickle
from pathlib import Path

from gensim.models.word2vec import Text8Corpus, LineSentence, Word2Vec
from nltk.corpus import movie_reviews, treebank

from train import data_dir


special_dir = data_dir / 'special'


def retrain(orig_model_name, sentences, corpus_name, iter=10):
    orig_model_path = special_dir / orig_model_name
    model = Word2Vec.load(orig_model_path.as_posix())
    nb_sentences = len(sentences)
    (special_dir / corpus_name).mkdir(exist_ok=True)
    for i in range(1, iter + 1):
        dest_name = "{}_{}_{}".format(orig_model_name, corpus_name, i)
        dest_path = special_dir / corpus_name / dest_name
        model.train(sentences, total_examples=nb_sentences)
        model.save(dest_path.as_posix())


def compute_accuracy(results_path, corpus_name):
    results_dir = Path(results_path)
    results_paths = results_dir.glob(corpus_name + '_*')
    results = [(int(p.name.split('_')[-1]), p) for p in results_paths]
    results.sort()
    yield "train_count, " + corpus_name
    for i, fpath in results:
        with fpath.open('rb') as f:
            data = pickle.load(f)
        accuracies = [
            len(sect['correct']) / (len(sect['correct']) + len(sect['incorrect']))
            for sect in data if len(sect['correct']) + len(sect['incorrect']) > 0
        ]
        average = sum(accuracies) / len(accuracies)
        yield "{}, {:.2%}".format(i, average)


def all_accuracies():
    tests = [
        ('treebank_accu_general', 'text8'),
        ('treebank_accu_special', 'text8'),
        ('treebank_accu_general', 'wiki'),
        ('treebank_accu_special', 'wiki'),
        ('movie_reviews_accu_general', 'text8'),
        ('movie_reviews_accu_special', 'text8'),
        ('movie_reviews_accu_general', 'wiki'),
        ('movie_reviews_accu_special', 'wiki'),
    ]
    for result, corpus in tests:
        path = special_dir / result
        data = compute_accuracy(path, corpus)
        print("==============", result, sep='\n')
        print(*data, sep='\n')
        print()


def main():
    # retrain('text8', movie_reviews.sents(), 'movie_reviews')
    # retrain('wiki', movie_reviews.sents(), 'movie_reviews')

    all_accuracies()


if __name__ == '__main__':
    main()
