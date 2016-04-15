# import modules & set up logging
import gensim
import os
import pickle
import logging


def evaluate_questions(questions, modeldir, outdir):
    for filename in os.listdir(modeldir):
        # Check if file can and should be evaluated
        if not filename.startswith('.') and filename not in os.listdir(outdir) and not filename.endswith('.npy'):
            logging.info('Computing accuracy of model: ' + filename)
            model = gensim.models.Word2Vec.load(modeldir + filename)
            sections = model.accuracy(questions)
            # Write out to pickle file
            outfile = open(outdir + filename, 'wb+')
            pickle.dump(sections, outfile)
            outfile.close()


def run_evals():
    logging.getLogger().setLevel(logging.INFO)

    evaluate_questions('../word2vec_data/questions-words.txt', '../word2vec_data/results_cbow/', '../word2vec_data/eval_cbow/')

    evaluate_questions('../word2vec_data/brands_questions.txt', '../word2vec_data/results_cbow/',
                       '../word2vec_data/eval_cbow_brands/')

    evaluate_questions('../word2vec_data/questions-words.txt', '../word2vec_data/results_sg/',
                       '../word2vec_data/eval_sg/')

    evaluate_questions('../word2vec_data/brands_questions.txt', '../word2vec_data/results_sg/',
                       '../word2vec_data/eval_sg_brands/')

def section_accuracy(section):
    correct, incorrect = section['correct'], section['incorrect']
    if correct + incorrect > 0:
        return correct / (correct + incorrect)



def main():
    run_evals()


if __name__ == '__main__':
    main()
