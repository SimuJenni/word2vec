# import modules & set up logging
import gensim
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
import os
import pickle
import logging
import numpy as np

from six import iteritems, itervalues


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


def evaluate_questions_kNN(questions, modeldir, outdir):
    for filename in os.listdir(modeldir):
        # Check if file can and should be evaluated
        if not filename.startswith('.') and filename not in os.listdir(outdir) and not filename.endswith('.npy'):
            logging.info('Computing accuracy of model: ' + filename)
            model = gensim.models.Word2Vec.load(modeldir + filename)
            sections = kNN_accuracy(model, questions)
            # Write out to pickle file
            outfile = open(outdir + filename, 'wb+')
            pickle.dump(sections, outfile)
            outfile.close()


def kNN_accuracy(model, questions, k=100, restrict_vocab=30000, most_similar=gensim.models.Word2Vec.most_similar, use_lowercase=True):
    """
    Compute k-Nearest Neighbor accuracy of the model.
    Code copied and adjusted from gensim.Model.Word2Vec.accuracy.
    """
    ok_vocab = dict(sorted(iteritems(model.vocab),
                           key=lambda item: -item[1].count)[:restrict_vocab])
    ok_index = set(v.index for v in itervalues(ok_vocab))

    sections, section = [], None

    found = np.zeros(k)     # Stores which of the k neighbors matched how many words
    section_count = 0
    for line_no, line in enumerate(utils.smart_open(questions)):
        line = utils.to_unicode(line)
        if line.startswith(': '):
            # a new section starts => store the old section
            if section:
                accuracy = np.cumsum(found)/section_count
                section['accuracy'] = accuracy
                sections.append(section)
                logging.info("finished section: %s" % section['section'])
            section = {'section': line.lstrip(': ').strip(), 'accuracy': []}
            section_count = 0   # Resetting count
            found = np.zeros(k)
        else:
            if not section:
                raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
            try:
                if use_lowercase:
                    a, b, c, expected = [word.lower() for word in
                                         line.split()]  # assumes vocabulary preprocessing uses lowercase, too...
                else:
                    a, b, c, expected = [word for word in line.split()]
            except:
                logging.info("skipping invalid line #%i in %s" % (line_no, questions))
            if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                logging.debug("skipping line #%i with OOV words: %s" % (line_no, line.strip()))
                continue

            section_count = section_count + 1
            ignore = set(model.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
            predicted = None
            # find the most likely prediction, ignoring OOV words and input words
            sims = most_similar(model, positive=[b, c], negative=[a], topn=False, restrict_vocab=restrict_vocab)
            sortedSims = matutils.argsort(sims, reverse=True)
            count = 0
            for index in matutils.argsort(sims, reverse=True):
                if index in ok_index and index not in ignore:
                    predicted = model.index2word[index]
                    if predicted != expected:
                        count = count + 1
                        if count==k:
                            break
                    else:
                        found[count] = found[count] + 1

    if section:
        # store the last section, too
        accuracy = np.cumsum(found) / section_count
        section['accuracy'] = accuracy
        sections.append(section)

    total_acc = sum((s['accuracy'] for s in sections))/len(sections)
    total = {
        'section': 'total',
        'accuracy': total_acc,
    }

    logging.info("finished section: %s" , total['section'])
    logging.info("total accuracy: %s" , total['accuracy'])

    sections.append(total)
    return sections


def main():
    logging.getLogger().setLevel(logging.INFO)
    # run_evals()
    evaluate_questions_kNN('../word2vec_data/questions-words.txt',
                           '../word2vec_data/models/',
                           '../word2vec_data/accuracy/')

if __name__ == '__main__':
    main()
