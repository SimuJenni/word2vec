# import modules & set up logging
import gensim, logging
from gensim.models.word2vec import Text8Corpus
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus = Text8Corpus('../word2vec_data/text8')

# Train baseline model
model = gensim.models.Word2Vec(corpus, workers=16)
model.save('../word2vec_data/results/baseline')