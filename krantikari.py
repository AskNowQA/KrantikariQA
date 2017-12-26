"""
    Author: geraltofrivia; saist1993
    The objective of this file to implement Krantikari QA.

    NOTE: A lot of code scavenged from preProcessing.py, some from parser.py


    Needs:
        - trained Keras model
        - working dbpedia interface
        - question
        - topic entity

    Pseudocode:
            +------------------------------------------------------------------------+
            |                  get a question                                        |
            |                       +                                                |
            |                       |                                                |
            |                       v                                                |
            |                  get entities                                          |
            |                   +         +                                          |
            |                   |         |                                          |
            |      1 entity  <--+         +-->   2 entities                          |
            |            +                              +                            |
            |            |                              |                            |
            |            v                              v                            |
            |      get subgraph                   get subgraph for  - S_e1           |
            |            +                            first entity                   |
            |            |                                  +                        |
            |            v                                  |                        |
            |    generate 1-hop paths                       v                        |
            |       ENT1 +/- R1                   get subgraph for  - S_e1           |
            |            |                           second entity                   |
            |            v                                  +                        |
            |     Pick top k options                        |                        |
            |            +                                  v                        |
            |            |                     Intersect them - S = S_e1 AND S_e2    |
            |            v                                  +                        |
            |    generate 2-hop paths                       |                        |
            |     ENT1 +/- R1 +/- R2                        v                        |
            |            |                       Generate paths in S like:           |
            |            v                       E1 +- R1 +- R2 +- E2                |
            |     Rank, pick top k                          +                        |
            |            +                                  |                        |
            |            |                                  v                        |
            |            |                         Rank, pick top k                  |
            |            |                                  +                        |
            |            |                                  |                        |
            |            |                                  |                        |
            |            |                                  |                        |
            |            +->  post ranking filters   <------+                        |
            |                        +                                               |
            |                        |                                               |
            |                        v                                               |
            |             convert core chain to sparql                               |
            |                        +                                               |
            |                        |                                               |
            |                        v                                               |
            |       detect if RDF:type constraints are needed                        |
            |                        +                                               |
            |                        |                                               |
            |                        v                                               |
            |                      ANSWER                                            |
            |                                                                        |
            +------------------------------------------------------------------------+


    Current Version: just one topic entity, multiple hops @TODO: Keep updating this.
"""

# Imports
import os
import json
import random
import pickle
import warnings
import traceback
import numpy as np
from pprint import pprint
from keras.models import load_model

# Local file imports
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils


# Some MACROS
DEBUG = True
MODEL_DIR = 'data/training/multi_path_mini/model_00/model.h5'
glove_location = \
    {
        'dir': "./resources",
        'raw': "glove.6B.300d.txt",
        'parsed': "glove_parsed_small.pickle"
    }


# Global Variables
dbp = db_interface.DBPedia(_verbose=True, caching=False)    # Summon a DBpedia interface
model = None                                                # Keras model to be used for ranking
embedding_glove = {}


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def vectorize(_tokens, _report_unks=False):
    """
        Function to embed a sentence and return it as a list of vectors.
        WARNING: Give it already split. I ain't splitting it for ye.

        :param _tokens: The sentence you want embedded. (Assumed pre-tokenized input)
        :param _report_unks: Whether or not return the out of vocab words
        :return: Numpy tensor of n * 300d, [OPTIONAL] List(str) of tokens out of vocabulary.
    """

    op = []
    unks = []
    for token in _tokens:

        # Small cap everything
        token = token.lower()

        try:
            token_embedding = embedding_glove[token]

        except KeyError:
            if _report_unks: unks.append(token)
            token_embedding = np.zeros(300, dtype=np.float32)

        op += [token_embedding]

    return (np.asarray(op), unks) if _report_unks else np.asarray(op)


def prepare():
    """
        **Call this function prior to doing absolutely anything else.**

        :param None
        :return: None
    """
    global embedding_glove

    if DEBUG: print("Loading Glove.")

    try:
        embedding_glove = pickle.load(open(os.path.join(glove_location['dir'], glove_location['parsed'])))
    except IOError:
        # Glove is not parsed and stored. Do it.
        if DEBUG: warnings.warn(" GloVe is not parsed and stored. This will take some time.")

        embedding_glove = {}
        f = open(os.path.join(glove_location['dir'], glove_location['raw']))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_glove[word] = coefs
        f.close()

        # Now convert this to a numpy object
        pickle.dump(embedding_glove, open(os.path.join(glove_location['dir'], glove_location['parsed']), 'w+'))

        if DEBUG: print("GloVe successfully parsed and stored. This won't happen again.")


def convert_core_chain_to_sparql(_core_chain):
    pass


def similar_predicates(_question, _predicates, _k=5):
    """
        Function used to tokenize the question and compare the tokens with the predicates.
        Then their top k are selected.

    :param _question: a string of question
    :param _predicates: a list of strings (surface forms/uri? @TODO: Verify)
    :param _k: int :- select top k from sorted list
    :return: a list of strings (subset of predicates)
    """
    # Tokenize question
    qt = nlutils.tokenize(_question, _remove_stopwords=True)

    # Vectorize question
    v_qt = vectorize(qt)

    # Vectorize predicates
    v_pt = np.asarray([ np.mean(vectorize(nlutils.tokenize(x)), axis=0) for x in _predicates])

    # Compute similarity
    similarity_mat = np.dot(v_pt, np.transpose(v_qt))

    # Find the best scoring values for every path
    argmaxes = np.argsort(np.max(similarity_mat, axis=1))

    # Use this to choose from _predicates and return @TODO.


def runtime(_question, _entities, _return_core_chains = False, _return_answers = False):
    """
        This function inputs one question, and topic entities, and returns a SPARQL query (or s'thing else)

    :param _question: a string of question
    :param _entities: a list of strings (each being a URI)
    :return: SPARQL/CoreChain/Answers (and or)
    """

    # Algo differs based on whether there's one topic entity or two

    if len(_entities) == 1:

        # Get 1-hop subgraph around the entity
        right_properties, left_properties = dbp.get_properties(_uri=_entities[0], label=False)

        # Get the surface form of these properties
        # @TODO

        # Generate 1-hop paths based on word2vec similarity
        right_properties_filtered = similar_predicates( _question=_question, _predicates=right_properties, k=5)
        left_properties_filtered = similar_predicates( _question=_question, _predicates=left_properties, k=5)

    if len(_entities) >= 2:
        pass




