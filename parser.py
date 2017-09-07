"""
    Author: geraltofrivia

    This script takes the json created by the pre-processing module and converts them into X and Y for the network.
    This X and Y depend on the network architecture that we're following so is expected to change now and then.

    Done:
        -> Embed a sentence

"""
import os
import pickle
import warnings
import numpy as np

from pprint import pprint
from gensim import models

# Some Macros
WORD2VEC_DIR = "./resources"  # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
GLOVE_DIR = "./resources"  # https://nlp.stanford.edu/projects/glove/
EMBEDDING = "GLOVE"  # OR WORD2VEC
EMBEDDING_DIM = 300
DEBUG = True
embedding_glove, embedding_word2vec = {}, {}  # Declaring the two things we're gonna use


# Better warning formatting. Ignore
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = better_warning
if DEBUG: warnings.warn(" DEBUG macro is enabled. Expect cluttered console!")

def prepare(_embedding = EMBEDDING):
    """
        **Call this function prior to doing absolutely anything else.**

        :param
            _embedding: str | either GLOVE or WORD2VEC.
                Choose which one to use.
        :return: None
    """
    global embedding_glove, embedding_word2vec

    # Preparing embeddings.
    if EMBEDDING == "GLOVE":

        if DEBUG: print "Using Glove."

        try:
            embedding_glove = pickle.load(open(os.path.join(GLOVE_DIR, "glove_parsed.pickle")))
        except IOError:
            # Glove is not parsed and stored. Do it.
            if DEBUG: warnings.warn(" GloVe is not parsed and stored. This will take some time.")

            embedding_glove = {}
            f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'))
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_glove[word] = coefs
            f.close()

            # Now convert this to a numpy object
            pickle.dump(embedding_glove, open(os.path.join(GLOVE_DIR, "glove_parsed.pickle"), 'w+'))

            if DEBUG: print "GloVe successfully parsed and stored. This won't happen again."

    elif EMBEDDING == "WORD2VEC":
        if DEBUG: print "Using Glove."

        embedding_word2vec = models.KeyedVectors.load_word2vec_format(
            os.path.join(WORD2VEC_DIR, 'GoogleNews-vectors-negative300.bin'), binary=True)


def sentence_embedding(_input, _report_unks = False):
    """
        Function to embed a sentence and return it as a list of vectors.
        Eg. sentence = a list of words ["The fastest person on the planet."]

        Changes:
            - removes question marks
            - removes commas
            - removes trailing spaces
    """

    # Cleaned sentence
    cleaned_input = _input.replace("?", "").replace(",", "").strip()

    # Split the sentence into word tokens
    # @TODO: Use a proper tokenizer.
    tokens = cleaned_input.split()

    # Logic for Glove
    op = []
    unks = []
    for token in tokens:
        try:
            if EMBEDDING == "GLOVE": token_embedding = embedding_glove[token]
            elif EMBEDDING == 'WORD2VEC': token_embedding = embedding_word2vec[token]
        except KeyError:
            if _report_unks: unks.append(token)
            token_embedding = np.zeros(300, dtype=np.float32)
        op += [token_embedding]
    return np.asarray(op) if _report_unks else np.asarray(op), unks


# Prepare X and Y
# @TODO

if __name__ == "__main__":
    """
        Tests:
            1. Load an embedding
            2. Check it for some sentences
            3. Fire up a loop for token tests
            4. Run it on LC QuAD and report the number of questions with unks and total unks encountered.
    """
    options = raw_input("Type in the numbers corresponding to the tests you want to run.")

    if "1" in options:
        prepare("GLOVE")
        print "Finished loading the embedding. Moving on."

    if "2" in options:
        sents = [
            "Who is the president of United States?",
            "For which band does Joe Hahn perform?",
            "Name some people playing for the Turkish handball league?"
        ]

        for sentence in sents:
            embedding = sentence_embedding(sentence)
            pprint(embedding)
            raw_input(sentence)

    if "3" in options:
        while True:
            tok = raw_input("Enter your word")
            if tok.lower() in ["stop", "bye", "q", "quit", "exit"]: break
            print sentence_embedding(tok)

    if "4" in options:
        # @TODO: implement this.
        pass



