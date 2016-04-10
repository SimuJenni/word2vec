# import modules & set up logging
import gensim
import os
import pickle
import logging

questions = '../word2vec_data/questions-words.txt'
resultsDir = '../word2vec_data/results/'
evalDir = '../word2vec_data/eval/'

logging.getLogger().setLevel(logging.INFO)

for filename in os.listdir(resultsDir):
    # Check if file can and should be evaluated
    if not filename.startswith('.') and filename not in os.listdir(evalDir):
        logging.info('Computing accuracy of model: ' + filename)
        model = gensim.models.Word2Vec.load(resultsDir + filename)
        sections = model.accuracy(questions)
        # Write out to pickle file
        outfile = open(evalDir + filename, 'w+')
        pickle.dump(sections, outfile)
        outfile.close()

