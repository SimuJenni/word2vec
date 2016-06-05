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


def most_similar_dice(model, positive=[], negative=[], topn=10, restrict_vocab=None):
    """
    Find the top-N most similar words. Positive words contribute positively towards the
    similarity, negative words negatively.
    This method computes dice similarity between a simple mean of the projection
    weight vectors of the given words and the vectors for each word in the model.
    Method copied and adjusted from gensim.models.word2vec.most_similar
    """
    model.init_sims()

    if isinstance(positive, string_types) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in positive
        ]
    negative = [
        (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in negative
        ]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, ndarray):
            mean.append(weight * word)
        elif word in model.vocab:
            # mean.append(weight * model.syn0norm[self.vocab[word].index])
            mean.append(weight * model.syn0[self.vocab[word].index]) # Probably have to use syn0 instead of syn0norm to get the unnormalized vectors
            all_words.add(model.vocab[word].index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    # mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
    mean = array(mean).mean(axis=0)

    #limited = model.syn0norm if restrict_vocab is None else model.syn0norm[:restrict_vocab]
    limited = model.syn0 if restrict_vocab is None else model.syn0[:restrict_vocab]
    #dists = dot(limited, mean)
    dists = abs(dot(limited, mean))/(abs(limited) + abs(mean))
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # ignore (don't return) words from the input
    result = [(model.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
    return result[:topn]


def main():
    logging.getLogger().setLevel(logging.INFO)
    run_evals()
    #evaluate_questions_kNN('../word2vec_data/questions-words.txt',
                           # '../word2vec_data/models/',
                           # '../word2vec_data/accuracy/')

if __name__ == '__main__':
    main()
