# import modules & set up logging
import gensim, logging
from gensim.models.word2vec import Text8Corpus
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus = Text8Corpus('../word2vec_data/text8')
model = gensim.models.Word2Vec(corpus, size=200, window=8, min_count=5, workers=24, sample=1e-4, iter=15, negative=25, hs=0)

model.save('..word2vec_data/results/mymodel')