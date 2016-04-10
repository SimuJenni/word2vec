# import modules & set up logging
import gensim, logging
from gensim.models.word2vec import Text8Corpus
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus = Text8Corpus('../word2vec_data/text8')

# Train models with different learning rates
windows = range(2, 12)

for w in windows:
    model = gensim.models.Word2Vec(corpus, window=w, workers=16)
    model.save('../word2vec_data/results/window_'+str(w))
