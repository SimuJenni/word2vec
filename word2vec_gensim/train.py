import logging
import sys
from pathlib import Path

import gensim
from gensim.models.word2vec import Text8Corpus


data_dir = Path('../word2vec_data')
corpus = Text8Corpus('../word2vec_data/text8')
# import ipdb; ipdb.set_trace()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Train models with different learning rates
alphas = list(4**x for x in range(-4, 6))

# for a in alphas:
#     model = gensim.models.Word2Vec(corpus, alpha=a, workers=16)
#     model.save('../word2vec_data/results/alpha_'+str(a))


def setup():
    if not data_dir.exists():
        logging.fatal('No "../word2vec_data" folder!')
        logging.fatal('Please create it or run the script from the right directory.')
        sys.exit(1)
    for folder in ('results', 'eval'):
        (data_dir / folder).mkdir(exist_ok=True)


def run_trainings(corpus, param, values):
    corpus_name = Path(corpus.fname).name
    default_params = dict(sentences=corpus, workers=16, iter=10)

    for val in values:
        fname = '{}_{}_{}'.format(corpus_name, param, val)
        dest_path = data_dir / 'results' / fname
        if dest_path.exists():
            logging.info("'{}' already exists, don't train again".format(fname))
            continue

        logging.info("Start training '{}'".format(fname))
        params = default_params.copy()
        params[param] = val
        model = gensim.models.Word2Vec(**params)
        model.save(dest_path.as_posix())




def main():
    setup()


if __name__ == '__main__':
    main()
