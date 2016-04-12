# import modules & set up logging
import gensim
import os
import pickle
import logging


def evaluate_questions(questions, modeldir, outdir):
    for filename in os.listdir(modeldir):
        # Check if file can and should be evaluated
        if not filename.startswith('.') and filename not in os.listdir(outdir):
            logging.info('Computing accuracy of model: ' + filename)
            model = gensim.models.Word2Vec.load(modeldir + filename)
            sections = model.accuracy(questions)
            # Write out to pickle file
            outfile = open(outdir + filename, 'w+')
            pickle.dump(sections, outfile)
            outfile.close()

logging.getLogger().setLevel(logging.INFO)

modelDir = '../word2vec_data/results/'

evaluate_questions('../word2vec_data/questions-words.txt', modelDir, '../word2vec_data/eval/')

evaluate_questions('../word2vec_data/questions-words.txt', modelDir, '../word2vec_data/eval_brands/')
