# import modules & set up logging
import gensim, logging
from gensim.models.word2vec import Text8Corpus
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus = Text8Corpus('../word2vec_data/text8')

# Train models with different learning rates
alphas = list(4**x for x in range(-4, 6))

for a in alphas:
    model = gensim.models.Word2Vec(corpus, alpha=a)
    model.save('../word2vec_data/results/alpha_'+str(a))
