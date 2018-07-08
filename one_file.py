
from __future__ import absolute_import

# Generic imporrts
import os
import sys
import json
import math
import pickle
import warnings
import traceback
import numpy as np
import pandas as pd
from progressbar import ProgressBar

# Keras Imports
from keras import optimizers
from keras.models import load_model
import keras.backend.tensorflow_backend as K
from keras.preprocessing.sequence import pad_sequences

# Local imports
import network as n
import data_preparation_rdf_type as drt
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils
from utils import prepare_vocab_continous as vocab_master


# Some Macros
DEBUG = True
MAX_SEQ_LENGTH = 25


# NN Macros
EPOCHS = 300
BATCH_SIZE = 880 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
NEGATIVE_SAMPLES = 1000
CANDIDATES_SPACE = None
LOSS = 'categorical_crossentropy'
OPTIMIZER = optimizers.Adam(LEARNING_RATE)

# Model Directories
DATASET = 'lcquad'
MULTI_HOP = False
MULTI_HOP_NUMBER = 5
CORECHAIN_MODEL_NAME = 'birnn_dot'
ID_BIG_DATA_FILENAME = 'id_big_data.json' if DATASET is 'lcquad' else 'qald_id_big_data.json'
DENIS_FILE_LOCATION = 'drogon:8001/slotptr20/test.out'
DENIS = False
POINTWISE = False
# #@lc-quad
# CORECHAIN_MODEL_DIR = './data/models/core_chain/%(model)s/%(data)s/model_17/model.h5' % {'model':CORECHAIN_MODEL_NAME, 'data': DATASET}
# RDFCHECK_MODEL_DIR = './data/models/rdf/%(data)s/model_04/model.h5' % {'data':DATASET}
# RDFEXISTENCE_MODEL_DIR = './data/models/type_existence/%(data)s/model_03/model.h5' % {'data':DATASET}
# INTENT_MODEL_DIR = './data/models/intent/%(data)s/model_03/model.h5' % {'data':'lcquad'}


# @sda-srv04
# CORECHAIN_MODEL_DIR = './data/models/core_chain/cnn_dense_dense/lcquad/model_00/model.h5'
#
CORECHAIN_MODEL_DIR = './data/models/core_chain/%(model)s/%(data)s/model_05/model.h5' % {'model' :CORECHAIN_MODEL_NAME, 'data': DATASET}
# CORECHAIN_MODEL_DIR = './data/models/core_chain/%(model)s/%(data)s/model_00/model.h5' % {'model':CORECHAIN_MODEL_NAME, 'data': DATASET}
RDFCHECK_MODEL_DIR = './data/models/rdf/%(data)s/model_01/model.h5' % {'data' :DATASET}
RDFEXISTENCE_MODEL_DIR = './data/models/type_existence/%(data)s/model_00/model.h5' % {'data' :DATASET}
INTENT_MODEL_DIR = './data/models/intent/%(data)s/model_00/model.h5' % {'data' :'lcquad'}

RELATIONS_LOC = os.path.join(n.COMMON_DATA_DIR, 'relations.pickle')
RDF_TYPE_LOOKUP_LOC = 'data/data/common/rdf_type_lookup.json'

# Configure at every run!
GPU = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

# Set the seed to clamp the stochastic nature.
np.random.seed(42) # Random train/test splits stay the same between runs

# Global variables

# Global variables
dbp = db_interface.DBPedia(_verbose=True, caching=True)
vocab, vectors = vocab_master.load()

reverse_vocab = {}
for keys in vocab:
    reverse_vocab[vocab[keys]] = keys

"""
    SPARQL Templates to be used to reconstruct SPARQLs from query graphs
"""
sparql_1hop_template = {
    "-": '%(ask)s %(count)s WHERE { { ?uri <http://dbpedia.org/property/%(r1)s> <%(te1)s> } UNION '
         + '{ ?uri <http://dbpedia.org/ontology/%(r1)s> <%(te1)s> } UNION '
         + '{ ?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s> }. %(rdf)s }',
    "+": '%(ask)s %(count)s WHERE { { <%(te1)s> <http://dbpedia.org/property/%(r1)s> ?uri } UNION '
         + '{ <%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?uri } UNION'
         + '{ ?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s> } . %(rdf)s }',
    "-c": '%(ask)s %(count)s WHERE { ?uri <%(r1)s> <%(te1)s> . %(rdf)s }',
    "+c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?uri . %(rdf)s }',
}
sparql_boolean_template = {
    "+": '%(ask)s %(count)s WHERE { { <%(te1)s> <http://dbpedia.org/property/%(r1)s> <%(te2)s> } UNION '
         + '{ <%(te1)s> <http://dbpedia.org/ontology/%(r1)s> <%(te2)s> } UNION '
         + '{ ?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s> } . %(rdf)s }'
    # "": '%(ask)s %(count)s WHERE { { <%(te1)s> <http://dbpedia.org/property/%(r1)s> <%(te2)s> } UNION '
    #                              + '{ <%(te1)s> <http://dbpedia.org/ontology/%(r1)s> <%(te2)s> } . %(rdf)s }',
}
sparql_2hop_1ent_template = {
    "++": '%(ask)s %(count)s WHERE { {<%(te1)s> <http://dbpedia.org/property/%(r1)s> ?x} UNION '
          + '{<%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?x} UNION '
          + '{<%(te1)s> <http://purl.org/dc/terms/%(r1)s> ?x} . '
          + '{?x <http://dbpedia.org/property/%(r2)s> ?uri} UNION'
          + '{?x <http://dbpedia.org/ontology/%(r2)s> ?uri} UNION'
          + '{?x <http://purl.org/dc/terms/%(r2)s> ?uri} . %(rdf)s }',
    "-+": '%(ask)s %(count)s WHERE { {?x <http://dbpedia.org/property/%(r1)s> <%(te1)s>} UNION '
          + '{?x <http://dbpedia.org/ontology/%(r1)s> <%(te1)s>} UNION '
          + '{?x <http://purl.org/dc/terms/%(r1)s> <%(te1)s>} .'
          + '{?x <http://dbpedia.org/property/%(r2)s> ?uri} UNION '
          + '{?x <http://dbpedia.org/ontology/%(r2)s> ?uri} UNION '
          + '{?x <http://purl.org/dc/terms/%(r2)s> ?uri} . %(rdf)s }',
    "--": '%(ask)s %(count)s WHERE { {?x <http://dbpedia.org/property/%(r1)s> <%(te1)s>} UNION '
          + '{?x <http://dbpedia.org/ontology/%(r1)s> <%(te1)s>} UNION '
          + '{?x <http://purl.org/dc/terms/%(r1)s> <%(te1)s>} .'
          + '{?uri <http://dbpedia.org/property/%(r2)s> ?x} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> ?x} UNION '
          + '{?uri <http://purl.org/dc/terms/%(r2)s> ?x} . %(rdf)s }',
    "+-": '%(ask)s %(count)s WHERE { {<%(te1)s> <http://dbpedia.org/property/%(r1)s> ?x} UNION '
          + '{<%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?x} UNION '
          + '{<%(te1)s> <http://purl.org/dc/terms/%(r1)s> ?x} .'
          + '{?uri <http://dbpedia.org/property/%(r2)s> ?x} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> ?x} UNION '
          + '{?uri <http://purl.org/dc/terms/%(r2)s> ?x} . %(rdf)s }',
    "++c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?x . ?x <%(r2)s> ?uri . %(rdf)s }',
    "-+c": '%(ask)s %(count)s WHERE { ?x <%(r1)s> <%(te1)s> . ?x <%(r2)s> ?uri . %(rdf)s }',
    "--c": '%(ask)s %(count)s WHERE { ?x <%(r1)s> <%(te1)s> . ?uri <%(r2)s> ?x . %(rdf)s }',
    "+-c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?x . ?uri <%(r2)s> ?x . %(rdf)s }'
}

sparql_2hop_2ent_template = {
    "+-": '%(ask)s %(count)s WHERE { {<%(te1)s> <http://dbpedia.org/property/%(r1)s> ?uri} UNION '
          + '{<%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?uri} UNION '
          + '{<%(te1)s> <http://purl.org/dc/terms/%(r1)s> ?uri} . '
          + '{<%(te2)s> <http://dbpedia.org/property/%(r2)s> ?uri} UNION '
          + '{<%(te2)s> <http://dbpedia.org/ontology/%(r2)s> ?uri} UNION'
          + '{<%(te2)s> <http://purl.org/dc/terms/%(r2)s> ?uri} . %(rdf)s }',
    "--": '%(ask)s %(count)s WHERE { {?uri <http://dbpedia.org/property/%(r1)s> <%(te1)s>} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r1)s> <%(te1)s>} UNION '
          + '{?uri <http://purl.org/dc/terms/s> <%(te1)s>} . '
          + '{?uri <http://dbpedia.org/property/%(r2)s> <%(te2)s>} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> <%(te2)s>} UNION '
          + '{?uri <http://purl.org/dc/terms/%(r2)s> <%(te2)s>} . %(rdf)s }',
    "-+": '%(ask)s %(count)s WHERE { {?uri <http://dbpedia.org/property/%(r1)s> <%(te1)s>} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r1)s> <%(te1)s>} UNION'
          + '{?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s>} .'
          + '{?uri <http://dbpedia.org/property/%(r2)s> <%(te2)s>} UNION'
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> <%(te2)s>} UNION'
          + '{?uri <http://purl.org/dc/terms/%(r2)s> <%(te2)s>} . %(rdf)s }',
    "++": '%(ask)s %(count)s WHERE { {<%(te1)s> <http://dbpedia.org/property/%(r1)s> ?uri} UNION '
          + '{<%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?uri} UNION'
          + '{<%(te1)s> <http://purl.org/dc/terms/%(r1)s> ?uri} .'
          + '{?uri <http://dbpedia.org/property/%(r2)s> <%(te2)s>} UNION'
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> <%(te2)s>} UNION'
          + '{?uri <http://purl.org/dc/terms/%(r2)s> <%(te2)s>}  . %(rdf)s }',
    "+-c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?uri . <%(te2)s> <%(r2)s> ?uri . %(rdf)s }',
    "--c": '%(ask)s %(count)s WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <%(te2)s> . %(rdf)s }',
    "-+c": '%(ask)s %(count)s WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <%(te2)s> . %(rdf)s }',
    "++c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?uri . ?uri <%(r2)s> <%(te2)s> . %(rdf)s }',
}

rdf_constraint_template = ' ?%(var)s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <%(uri)s> . '

# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

def custom_loss(y_true, y_pred):
    """
        Max-Margin Loss
    """
    # y_pos = y_pred[0]
    # y_neg= y_pred[1]
    diff = y_pred[: ,-1]
    # return K.sum(K.maximum(1.0 - diff, 0.))
    return K.sum(diff)

def rank_precision(model, test_questions, test_pos_paths, test_neg_paths, neg_paths_per_epoch=100, batch_size=1000):
    max_length = test_questions.shape[-1]
    questions = np.reshape(np.repeat(np.reshape(test_questions,
                                                (test_questions.shape[0], 1, test_questions.shape[1])),
                                     neg_paths_per_epoc h +1, axis=1), (-1, max_length))
    pos_paths = np.reshape(test_pos_paths,
                           (test_pos_paths.shape[0], 1, test_pos_paths.shape[1]))
    neg_paths = test_neg_paths[:, np.random.randint(0, NEGATIVE_SAMPLES, neg_paths_per_epoch), :]
    all_paths = np.reshape(np.concatenate([pos_paths, neg_paths], axis=1), (-1, max_length))

    outputs = model.predict([questions, all_paths, np.zeros_like(all_paths)], batch_size=batch_size)[: ,0]
    outputs = np.reshape(outputs, (test_questions.shape[0], neg_paths_per_epoc h +1))

    precision = float(len(np.where(np.argmax(outputs, axis=1 )= =0)[0]) ) /outputs.shape[0]
    return precision

def rank_precision_runtime(model, id_q, id_tp, id_fps, batch_size=1000, max_length=50 ,rdf=False):
    '''
        A function to pad the data for the model, run model.predict on it and get the resuts.

    :param id_q: A 1D array of the question
    :param id_tp: A 1D array of the true path
    :param id_fps: A list of 1D arrays of false paths
    :param batch_size: int: the batch size the model expects
    :param max_length: int: size with which we pad the data
    :return: ?? @TODO
    '''

    # Create empty matrices
    question = np.zeros((len(id_fps ) +1, max_length))
    paths = np.zeros((len(id_fps ) +1, max_length))

    # Fill them in
    question[: ,:id_q.shape[0]] = np.repeat(id_q[np.newaxis ,:min(id_q.shape[0], question.shape[1])],
                                           question.shape[0], axis=0)
    paths[0, :id_tp.shape[0]] = id_tp
    for i in range(len(id_fps)):
        if len(id_fps[i]) > max_length:
            paths[ i +1 ,:min(id_fps[i].shape[0] ,question.shape[1])] = id_fps[i][:max_length]
        else:
            paths[ i +1 ,:min(id_fps[i].shape[0], question.shape[1])] = id_fps[i]
    # Pass em through the model
    if POINTWISE and not rdf:
        results = model.predict([question, paths], batch_size=batch_size)[: ,0]
    else:
        results = model.predict([question, paths, np.zeros_like(paths)], batch_size=batch_size)[: ,0]
    return results

def rank_precision_metric(neg_paths_per_epoch):
    def metric(y_true, y_pred):
        scores = y_pred[:, 0]
        scores = K.reshape(scores, (-1, neg_paths_per_epoc h +1))
        hits = K.cast(K.shape(K.tf.where(K.tf.equal(K.tf.argmax(scores, axis=1) ,0)))[0], 'float32')
        precision = hit s /K.cast(K.shape(scores)[0], 'float32')
        # precision = float(len(np.where(np.argmax(all_outputs, axis=1)==0)[0]))/all_outputs.shape[0]
        return precision
    return metric

def get_glove_embeddings():
    """
        Get to use the glove embedding as a local var.
        Prepare embeddings if not done already.
    :return: np mat
    """
    from utils.embeddings_interface import __check_prepared__
    __check_prepared__('glove')

    from utils.embeddings_interface import glove_embeddings
    return glove_embeddings

def cross_correlation(x):
    a, b = x
    tf = K.tf
    a_fft = tf.fft(tf.complex(a, 0.0))
    b_fft = tf.fft(tf.complex(b, 0.0))
    ifft = tf.ifft(tf.conj(a_fft) * b_fft)
    return tf.cast(tf.real(ifft), 'float32')

def rel_id_to_rel(rel, _relations):
    """


    :param rel:
    :param _relations: The relation lookup is inverse here
    :return:
    """
    occurrences = []
    for key in _relations:
        value = _relations[key]
        if np.array_equal(value[3] ,np.asarray(rel)):
            occurrences.append(value)
    # print occurrences
    if len(occurrences) == 1:
        return occurrences[0][0]
    else:
        '''
            prefers /dc/terms/subject' and then ontology over properties
        '''
        if 'terms/subject' in occurrences[0][0]:
            return occurrences[0][0]
        if 'terms/subject' in occurrences[1][0]:
            return occurrences[1][0]
        if 'property' in occurrences[0][0]:
            return occurrences[1][0]
        else:
            return occurrences[0][0]

def return_sign(sign):
    if sign == 2:
        return '+'
    else:
        return '-'

def id_to_path(path_id, vocab, relations, reverse_vocab, core_chain = True):
    '''


    :param path_id:  array([   3, 3106,    3,  647]) - corechain wihtout entity
    :param vocab: from continuous id space to discrete id space.
    :param relations: inverse relation lookup dictionary
    :return: paths
    '''

    # mapping from discrete space to continuous space.
    path_id = np.asarray([reverse_vocab[i] for i in path_id])

    # find all the relations in the given paths
    if core_chain:
        '''
            Identify the length. Is it one hop or two.
            The assumption is '+' is 2 and '-' is 3
        '''
        rel_length = 1
        if 2 in path_id[1:].tolist() or 3 in path_id[1:].tolist():
            rel_length = 2

        if rel_length == 2:
            sign_1 = path_id[0]
            try:
                index_sign_2 = path_id[1:].tolist().index(2) + 1
            except ValueError:
                index_sign_2 = path_id[1:].tolist().index(3) + 1
            rel_1 ,rel_2 = path_id[1:index_sign_2] ,path_id[index_sign_ 2 +1:]
            rel_1 = rel_id_to_rel(rel_1 ,relations)
            rel_2 = rel_id_to_rel(rel_2 ,relations)
            sign_2 = path_id[index_sign_2]
            path = [return_sign(sign_1) ,rel_1 ,return_sign(sign_2) ,rel_2]
            return path
        else:
            sign_1 = path_id[0]
            rel_1 = path_id[1:]
            rel_1 = rel_id_to_rel(rel_1 ,relations)
            path = [return_sign(sign_1) ,rel_1]
            return path
    else:
        variable = path_id[0]
        sign_1 = path_id[1]
        rel_1 = rel_id_to_rel(path_id[2:] ,relations)
        pass

def rdf_type_candidates(data ,path_id, vocab, relations, reverse_vocab, only_x, core_chain = True):
    '''
        Takes in path ID (continous IDs, not glove vocab).
        And return type candidates (in continous IDs)
            based on whether we want URI or X candidates
    :param data:
    :param path_id:
    :param vocab:
    :param relations:
    :param reverse_vocab:
    :param core_chain:
    :return:
    '''

    # @TODO: Only generate specific candidates
    data = data['parsed-data']
    path = id_to_path(path_id, vocab, relations, reverse_vocab, core_chain = True)
    sparql = drt.reconstruct(data['entity'], path, alternative=True)
    sparqls = drt.create_sparql_constraints(sparql)

    if len(data['entity']) == 2:
        sparqls = [sparqls[1]]
    if len(path) == 2:
        sparqls = [sparqls[1]]
    type_x, type_uri = drt.retrive_answers(sparqls)

    # # Remove the "actual" constraint from this list (so that we only create negative samples)
    # try:
    #     type_x = [x for x in type_x if x not in data['constraints']['x']]
    # except KeyError:
    #     pass
    #
    # try:
    #     type_uri = [x for x in type_uri if x not in data['constraints']['uri']]
    # except KeyError:
    #     pass

    type_x_candidates, type_uri_candidates = drt.create_valid_paths(type_x, type_uri)
    # return type_x_candidates
    # Convert them to continous IDs.
    for i in range(len(type_x_candidates)):
        for j in range(len(type_x_candidates[i])):
            try:
                type_x_candidates[i][j] = vocab[type_x_candidates[i][j]]
            except KeyError:
                '''
                    vocab[1] refers to unknow word.
                '''
                type_x_candidates[i][j] = vocab[1]
    for i in range(len(type_uri_candidates)):
        for j in range(len(type_uri_candidates[i])):
            try:
                type_uri_candidates[i][j] = vocab[type_uri_candidates[i][j]]
            except:
                type_uri_candidates[i][j] = vocab[1]
    # Return based on given input.
    return type_x_candidate s +type_uri_candidates
    # return type_x_candidates if only_x else type_uri_candidates

def load_relation():
    relations = pickle.load(open(RELATIONS_LOC))
    inverse_relations = {}
    for key in relations:
        value = relations[key]
        new_key = value[0]
        value[0] = key
        value.append(new_key)
        inverse_relations[new_key] = value

    return inverse_relations

def create_true_positive_rdf_path(data):
    '''
        Creates true rdf-type constraint, if it exists.
        :param data: One specific node of id_data.
        :return: None if no true rdf-type constraint; else returns rdf-type constraint in continuous id space.
    '''

    datum = data
    if '?x' in datum['parsed-data']['constraints'].keys():
        pos_path = "x + " + dbp.get_label(datum['parsed-data']['constraints']['?x'])
    elif '?uri' in datum['parsed-data']['constraints'].keys():
        pos_path = "uri + " + dbp.get_label(datum['parsed-data']['constraints']['?uri'])
    else:
        return None
    pos_path = embeddings_interface.vocabularize(nlutils.tokenize(pos_path))
    for i in range(0 ,len(pos_path)):
        pos_path[i] = vocab[pos_path[i]]
    return pos_path

# TODO: get
# Divide the data into diff blocks
split_point = lambda x: int(len(x) * .80)
def train_split(x):
    return x[:split_point(x)]
def test_split(x):
    return x[split_point(x):]


def rdf_type_check(question ,model_rdf_type_check, max_length = 30):
    """

    :param question: vectorize question
    :param model_rdf_type_check: model
    :return:
    """
    question_padded = np.zeros((1 ,max_length))
    try:
        question_padded[: ,:question.shape[0]] = question
    except ValueError:
        question_padded = question[: ,:question_padded.shape[0]]
    prediction = np.argmax(model_rdf_type_check.predict(question_padded))
    if prediction == 0:
        return True
    else:
        return False

def remove_positive_path(positive_path, negative_paths):
    new_negative_paths = []
    for i in range(0, len(negative_paths)):
        if not np.array_equal(negative_paths[i], positive_path):
            new_negative_paths.append(np.asarray(negative_paths[i]))
    return positive_path, np.asarray(new_negative_paths)

def load_reverse_rdf_type():
    rdf_type = json.load(open(RDF_TYPE_LOOKUP_LOC))
    rdf = {}
    for classes in rdf_type:
        rdf[classes] = embeddings_interface.vocabularize(nlutils.tokenize(dbp.get_label(classes)))
    return rdf

def convert_rdf_path_to_text(path):
    """
        Function used to convert a path (of continous IDs) to a text path.
        Eg. [ 5, 3, 420] : [uri, dbo:Poop]

    :param path: list of strings
    :return: list of text
    """

    # First we need to convert path to glove vocab
    path = [reverse_vocab[x] for x in path]

    # Then to convert this to text
    var = ''
    for key in embeddings_interface.glove_vocab.keys():
        if embeddings_interface.glove_vocab[key] == path[0]:
            var = key
            break

    dbo_class = ''
    for key in reverse_rdf_dict.keys():
        if list(reverse_rdf_dict[key]) == list(path[2:]):
            dbo_class = key
            break

    return [var, dbo_class]

def construct_paths(data ,relations ,qald=False):
    """
    :param data: a data node of id_big_data
    :return: unpadded , continous id spaced question, positive path, negative paths
    """
    question = np.asarray(data['uri']['question-id'])
    # questions = pad_sequences([question], maxlen=max_length, padding='post')

    # inverse id version of positive path and creating a numpy version
    positive_path_id = data['parsed-data']['path_id']
    false_positive_path = False
    if positive_path_id == [-1]:
        positive_path = np.asarray([-1])
        false_positive_path = True
    else:
        positive_path = []
        for path in positive_path_id:
            positive_path += [embeddings_interface.SPECIAL_CHARACTERS.index(path[0])]
            positive_path += relations[int(path[1:])][3].tolist()
        positive_path = np.asarray(positive_path)
    # padded_positive_path = pad_sequences([positive_path], maxlen=max_length, padding='post')

    # negative paths from id to surface form id
    negative_paths_id = data['uri']['hop-2-properties'] + data['uri']['hop-1-properties']
    negative_paths = []
    for neg_path in negative_paths_id:
        negative_path = []
        for path in neg_path:
            try:
                negative_path += [embeddings_interface.SPECIAL_CHARACTERS.index(path)]
            except ValueError:
                negative_path += relations[int(path)][3].tolist()
        negative_paths.append(np.asarray(negative_path))
    negative_paths = np.asarray(negative_paths)
    # negative paths padding
    # padded_negative_paths = pad_sequences(negative_paths, maxlen=max_length, padding='post')

    # explicitly remove any positive path from negative path
    positive_path, negative_paths = remove_positive_path(positive_path, negative_paths)

    # remap all the id's to the continous id space.

    # passing all the elements through vocab
    question = np.asarray([vocab[key] for key in question])
    if not false_positive_path:
        positive_path = np.asarray([vocab[key] for key in positive_path])
    for i in range(0, len(negative_paths)):
        # temp = []
        for j in xrange(0, len(negative_paths[i])):
            try:
                negative_paths[i][j] = vocab[negative_paths[i][j]]
            except:
                negative_paths[i][j] = vocab[0]
                # negative_paths[i] = np.asarray(temp)
                # negative_paths[i] = np.asarray([vocab[key] for key in negative_paths[i] if key in vocab.keys()])
    if qald:
        return question, positive_path, negative_paths ,false_positive_path
    return question ,positive_path ,negative_paths

def question_intent(model_question_intent ,padded_question):
    """
        predicting the intent of the question.
            List, Ask, Count.
    """
    intent = np.argmax(model_question_intent.predict(padded_question))
    return ['count', 'ask', 'list'][intent]

def rdf_constraint_check(padded_question ,model_rdf_type_existence):
    """
        predicting the existence of rdf type constraints.
    """
    return np.argmax(model_rdf_type_existence.predict(padded_question))

def pad_question(question):
    """

    :param question: continuous space id question
    :return: padded question
    """
    padded_question = np.zeros(MAX_SEQ_LENGTH)
    padded_question[:min(MAX_SEQ_LENGTH, len(question))] = question[:min(MAX_SEQ_LENGTH, len(question))]
    padded_question = padded_question.reshape((1, padded_question.shape[0]))
    return padded_question

plus_id, minus_id = None, None # These are vocab IDs
def reconstruct_corechain(_chain ,relations):
    """
        Expects a corechain made of continous IDs, and returns it in its text format (uri form)
        @TODO: TEST!
    :param _chain: list of ints
    :return: str: list of strs
    """
    global plus_id, minus_id

    # Find the plus and minus ID.
    if not (plus_id and minus_id):
        plus_id = embeddings_interface.vocabularize(['+'])
        minus_id = embeddings_interface.vocabularize(['-'])

    corechain_vocabbed = [reverse_vocab[key] for key in _chain]

    # Find the hop-length of the corechain
    length = sum([ 1 for id in corechain_vocabbed if id in [plus_id, minus_id]])

    if length == 1:

        # Just one predicate. Find its uri
        uri = rel_id_to_rel(corechain_vocabbed[1:] ,relations)
        sign = '+' if corechain_vocabbed[0] == plus_id else '-'
        signs = [sign]
        uris = [uri]

    elif length == 2:

        # Find the index of the second sign
        index_second_sign = None
        for i in range(1, len(corechain_vocabbed)):
            if corechain_vocabbed[i] in [plus_id, minus_id]:
                index_second_sign = i
                break

        first_sign = '+' if corechain_vocabbed[0] == plus_id else '-'
        second_sign = '+' if corechain_vocabbed[index_second_sign] == plus_id else '-'
        first_uri = rel_id_to_rel(corechain_vocabbed[1:index_second_sign] ,relations)
        second_uri = rel_id_to_rel(corechain_vocabbed[index_second_sig n +1:] ,relations)

        signs = [first_sign, second_sign]
        uris = [first_uri, second_uri]

    else:
        # warnings.warn("Corechain length is unexpected. Help!")
        return [], []

    return signs, uris

def query_graph_to_sparql(_graph):
    """
        Expects a dict containing:
            best_path,
            intent,
            rdf_constraint,
            rdf_constraint_type,
            rdf_best_path

        Returns a composted SPARQL.

        1. Convert everything to strings.

    :param _graph: (see above)
    :return: str: SPARQL.
    """
    sparql_value = {}

    # Find entities
    entities = _graph['entities']

    print _graph

    # Convert the corechain to glove embeddings
    corechain_signs, corechain_uris = reconstruct_corechain(_graph['best_path'] ,relations)

    # Construct the stuff outside the where clause
    sparql_value["ask"] = 'ASK' if _graph['intent'] == 'ask' else 'SELECT DISTINCT'
    if _graph['intent'] == 'count':
        sparql_value["count"] = 'COUNT(?uri)'
    elif _graph['intent'] == 'ask':
        sparql_value["count"] = ''
    else:
        sparql_value["count"] = '?uri'

    # Check if we need an RDF constraint.
    if _graph['rdf_constraint']:
        try:
            rdf_constraint_values = {}
            rdf_constraint_values['var'] = _graph['rdf_constraint_type']
            rdf_constraint_values['uri'] = convert_rdf_path_to_text(_graph['rdf_best_path'])[1]

            sparql_value["rdf"] = rdf_constraint_template % rdf_constraint_values
        except IndexError:
            sparql_value["rdf"] = ''

    else:
        sparql_value["rdf"] = ''

    # Find the particular template based on the signs

    """
        Start putting stuff in template.
        Note: if we're dealing with a count query, we append a 'c' to the query.
            This does away with dbo/dbp union and goes ahead with whatever came in the question.
    """
    signs_key = ''.join(corechain_signs)

    if _graph['intent'] == 'ask':
        # Assuming that there is only single triple ASK queries.
        sparql_template = sparql_boolean_template['+']
        sparql_value["te1"] = _graph['entities'][0]
        sparql_value["te2"] = _graph['entities'][1]
        sparql_value["r1"] = corechain_uris[0].split('/')[-1]

    elif len(signs_key) == 1:
        # Single hop, non boolean.
        sparql_template = sparql_1hop_template[signs_ke y +'c' if _graph['intent'] == 'count' else signs_key]
        sparql_value["te1"] = _graph['entities'][0]
        sparql_value["r1"] = corechain_uris[0] if _graph['intent'] == 'count' else corechain_uris[0].split('/')[-1]

    else:
        # Double hop, non boolean.

        sparql_value["r1"] = corechain_uris[0] if _graph['intent'] == 'count' else corechain_uris[0].split('/')[-1]
        sparql_value["r2"] = corechain_uris[1] if _graph['intent'] == 'count' else corechain_uris[1].split('/')[-1]

        # Check if entities are one or two
        if len(_graph['entities']) == 1:
            sparql_template = sparql_2hop_1ent_template[signs_ke y +'c' if _graph['intent'] == 'count' else signs_key]
            sparql_value["te1"] = _graph['entities'][0]
        else:
            sparql_template = sparql_2hop_2ent_template[signs_ke y +'c' if _graph['intent'] == 'count' else signs_key]
            sparql_value["te1"] = _graph['entities'][0]
            sparql_value["te2"] = _graph['entities'][1]

    # Now to put the magic together
    sparql = sparql_template % sparql_value

    return sparql

def ground_truth_intent(data):
    """
        Legend: 010: ask
                100: count
                001: list
    """

    data = data['parsed-data']['sparql_query'][:data['parsed-data']['sparql_query'].lower().find('{')]
    # Check for ask
    if u'ask' in data.lower():
        return 'ask'

    if u'count' in data.lower():
        return 'count'

    return 'list'


def ground_truth_rdf(data):
    datum = data
    if '?x' in datum['parsed-data']['constraints'].keys():
        return 'x' ,dbp.get_label(datum['parsed-data']['constraints']['?x'])
    elif '?uri' in datum['parsed-data']['constraints'].keys():
        return 'uri' ,dbp.get_label(datum['parsed-data']['constraints']['?uri'])
    else:
        return 'none' ,'none'


def sparql_answer(sparql):
    test_answer = []
    interface_test_answer = dbp.get_answer(sparql)
    for key in interface_test_answer:
        test_answer = test_answer + interface_test_answer[key]
    return list(set(test_answer))


def evaluate(test_sparql, true_sparql, type, ground_type):
    # @TODO: If the type of test and true are differnt code would return an error.
    """
        Fmeasure for ask and count are 0/1.
        Also assumes the variable to be always uri.
        :param test_sparql: SPARQL generated by the pipeline
        :param true_sparql: True SPARQL
        :param type: COUNT/ASK/LIST
        :return: f1,precision,recall
    """
    '''
        First evaluate based on type. If the type prediction is wrong. Don't proceded. The f,p,r =0
    '''
    if type != ground_type:
        return 0.0 ,0.0 ,0.0

    if type == "list":
        test_answer = sparql_answer(test_sparql)
        true_answer = sparql_answer(true_sparql)
        total_retrived_resutls = len(test_answer)
        total_relevant_resutls = len(true_answer)
        common_results = total_retrived_resutls - len(list(set(test_answer ) -set(true_answer)))
        if total_retrived_resutls == 0:
            precision = 0
        else:
            precision = common_result s *1. 0 /total_retrived_resutls
        if total_relevant_resutls == 0:
            recall = 0
        else:
            recall = common_result s *1. 0 /total_relevant_resutls
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = (2.0 * (precision * recall)) / (precision + recall)
        return f1 ,precision ,recall

    if type == "count":
        count_test = sparql_answer(test_sparql)
        count_true = sparql_answer(true_sparql)
        if count_test == count_true:
            return 1.0 ,1.0 ,1.0
        else:
            return 0.0 ,0.0 ,1.0

    if type == "ask":
        if dbp.get_answer(test_sparql) == dbp.get_answer(true_sparql):
            return 1.0 ,1.0 ,1.0
        else:
            return 0.0 ,0.0 ,1.0


def similarity(question ,path ,mode='mean'):
    '''
            Takes question and path mapped in continous space and computes similarity
            mode = avg/sum
    '''
    question_v = np.asarray([vectors[x] for x in question])
    path_v = np.asarray([vectors[x] for x in path])
    if mode == 'sum':
        question_avg = np.sum(question_v ,axis=0)
        path_avg = np.sum(path_v ,axis=0)
    else:
        question_avg = np.mean(question_v, axis=0)
        path_avg = np.mean(path_v, axis=0)
    if np.sum(path_v) == 0 or np.sum(question_v) == 0:
        return 0.0
    else:
        return np.dot(path_avg, question_avg) / (np.linalg.norm(question_avg) * np.linalg.norm(path_avg))


def prune_candidate_space(question ,paths ,k=None):
    sim = []
    for p in paths:
        sim.append(similarity(question ,p))
    if not k:
        return np.argsort(sim)
    if len(sim) > k:
        return np.argsort(sim)[-k:]
    else:
        return np.argsort(sim)

def convert(paths):
    paths_list = [p.tolist() for p in paths]
    first_hop_properties = []
    for p in paths_list:
        try:
            negative_index = p[1:].index(3)
            first_prop = p[:negative_inde x +1]
            first_hop_properties.append(first_prop)
        except ValueError:
            try:
                positive_index = p[1:].index(2)
                first_prop = p[:positive_index + 1]
                first_hop_properties.append(first_prop)
                continue
            except ValueError:
                first_hop_properties.append(p)
                continue
    # fhp = [np.asarray(f) for f in first_hop_properties]
    unique_fhp = []
    for p in first_hop_properties:
        if p not in unique_fhp:
            unique_fhp.append(p)
    # first_hop_properties = np.unique(np.asarray(first_hop_properties)).tolist()
    return unique_fhp


def dicta(paths ,fhps):
    paths_list = [p.tolist() for p in paths]
    picked_paths = []
    for fhp in fhps:
        l = len(fhp)
        for p in paths_list:
            try:
                if p[:l] == fhp:
                    picked_paths.append(p)
            except:
                continue
    return picked_paths

def search_string_to_rel(relations ,search_string):
    matches = []
    for r in relations:
        r_small_case = [j.lower() for j in relations[r][2]]
        if r_small_case == search_string:
            matches.append(r)
    if len(matches) == 1:
        return matches[0]
    else:
        best_match = []
        for m in matches:
            if "ontology" in relations[m][0]:
                best_match = m
                break
        else:
            best_match = matches[0]
        return best_match


def pred_to_rel(pred ,relations):
    if "EMPTY" in pred:
        pred = pred.replace('EMPTYEMPTYEMPTY' ,'')
        pred = pred.strip().split(' ')
        search_string = pred[1:]
        best_match = search_string_to_rel(relations ,search_string)
        matches = []
        rel_string = [pred[0] ,best_match]
    else:
        pred = pred.strip().split(' ')
        sign_1 = pred[0]
        '''
            find the index woith second sign
        '''
        try:
            index_sign_2 = pred[1:].index('-')
        except ValueError:
            index_sign_2 = pred[1:].index('+')
        rel_1, rel_2 = pred[1:index_sign_ 2 +1], pred[index_sign_2 + 2:]
        rel1 = search_string_to_rel(relations ,rel_1)
        rel2 = search_string_to_rel(relations ,rel_2)
        sign_2 = pred[index_sign_ 2 +1]
        rel_string = [sign_1 ,rel1 ,sign_2 ,rel2]
    return rel_string

def core_chain_accuracy(question ,paths ,positive_path ,negative_paths ,core_chain_counter ,model_corechain
                        ,no_positive_path ,printer=True):
    '''


    '''
    mrr = 0
    if no_positive_path and len(negative_paths) == 0:
        results.append([0.0, 0.0, 0.0])
        avg.append(0.0)
        print(sum(avg) * 1.0 / len(avg))
        # raw_input()
        if printer: print "in the first if condition"
        return question ,paths ,positive_path ,negative_paths ,core_chain_counter, '' ,mrr

    if len(negative_paths) == 0:
        best_path = positive_path
        core_chain_counter = core_chain_counter + 1
        mrr = 1
        if printer: print "in the second  if condition and mrr should be 1"
        return question, paths, positive_path, negative_paths, core_chain_counter, best_path ,mrr
    else:
        '''
            The output is made by stacking positive path over negative paths.
        '''
        if MULTI_HOP:
            fhp = convert(paths)
            fhp_numpy = [np.asarray(pa) for pa in fhp]
            fhp_numpy = np.asarray(fhp_numpy)

            output = rank_precision_runtime(model_corechain, question, fhp_numpy[0],
                                            fhp_numpy, 10000, MAX_SEQ_LENGTH)
            if MULTI_HOP_NUMBER < (len(output) - 1):
                indexes = np.argsort(output[1:])[-1 * (MULTI_HOP_NUMBER):]
            else:
                indexes = np.argsort(output[1:])


            print "at the numpy selection point"
            fhp = [fhp_numpy[ind].tolist() for ind in indexes]
            picked_paths = dicta(paths, fhp)
            picked_paths_numpy = [np.asarray(pa) for pa in picked_paths]
            picked_paths_numpy = np.asarray(picked_paths_numpy)

            output = rank_precision_runtime(model_corechain, question, picked_paths_numpy[0],
                                            picked_paths_numpy, 10000, MAX_SEQ_LENGTH)
            '''
                Find top k performing properties and then using it for the same
            '''
            paths = picked_paths_numpy
            '''
                MRR implementation is still not done.
            '''
            # if positive_path:
            #     index_positive_path = ''
            #     for p in range(len(paths)):
            #         if np.array_equal(p,positive_path):
            #             index_positive_path = p
            #     else:
            #         index_positive_path = ''
            best_path = paths[np.argmax(output[1:])]
        else:
            if no_positive_path:
                output = rank_precision_runtime(model_corechain, question, negative_paths[0],
                                                negative_paths, 10000, MAX_SEQ_LENGTH)
                best_path = negative_paths[np.argmax(output[1:])]
                mrr = 0
            else:
                output = rank_precision_runtime(model_corechain, question, positive_path, negative_paths, 10000, MAX_SEQ_LENGTH)

                best_path_index = np.argmax(output)
                mrr_output = np.argsort(output)[::-1]
                mrr_output = mrr_output.tolist()
                mrr = mrr_output.index(0) + 1
                if best_path_index == 0:
                    best_path = positive_path
                else:
                    best_path = negative_paths[best_path_index - 1]


        # best_path = paths[np.argmax(output[1:])]

        if not no_positive_path:
            if np.array_equal(best_path, positive_path):
                core_chain_counter = core_chain_counter + 1
        return question, paths, positive_path, negative_paths, core_chain_counter, best_path ,mrr



def construct_query_graph(data):
    '''

        The function takes in data node from id_big_data file and creates query grah which could then be used for evaluation.
    :return: query_graph
    '''

    try:
        print "in try loop"

    except:
        pass
# Some more globals
relations = load_relation()
reverse_relations = {}
for keys in relations:
    reverse_relations[relations[keys][0]] = [keys] + relations[keys][1:]
reverse_rdf_dict = load_reverse_rdf_type()

'''
    Core chain accuracy counter counts the number of time the core chain predicated is same as 
    positive path. This also includes for ask query.
    The counter might confuse the property and the ontology. 

    Similar functionality with rdf_type and intent
'''
core_chain_accuracy_counter = 0
intent_accuracy_counter = 0
rdf_type_existence_accuracy_counter = 0
query_graph_accuracy_counter = 0

'''
    c_flag  is true if the core_chain was correctly predicted. same is the case for i_flag and r_flag, rt_flag (correct candidate for rdf type)
'''
c_flag, i_flag, r_flag, rt_flag = False, False, False, False

'''
    Counts the number of times just using  word2vec similarity, the best path came the most similar. This will only work if
    CANDIDATE_SPACE is not none.
'''
word_vector_accuracy_counter = 0

'''
    Stores tuple of (fmeasure,precision,recall)
'''
results = []
'''
    keeps avg of fmeasure for each question. A real time fmeasure
'''
avg = []
MRR = []
counter = 0

Logging = {}

Logging['dataset'] = DATASET
Logging['CORECHAIN_MODEL_DIR'] = CORECHAIN_MODEL_DIR
Logging['RDFCHECK_MODEL_DIR'] = RDFCHECK_MODEL_DIR
Logging['RDFEXISTENCE_MODEL_DIR'] = RDFEXISTENCE_MODEL_DIR
Logging['INTENT_MODEL_DIR'] = INTENT_MODEL_DIR
Logging['main_log'] = []


if __name__ is "__main__":

    # Some more globals
    relations = load_relation()
    reverse_relations = {}
    for keys in relations:
        reverse_relations[relations[keys][0]] = [keys] + relations[keys][1:]
    reverse_rdf_dict = load_reverse_rdf_type()

    if DATASET == 'qald':
        # Load qald test data
        id_data_test = json.load \
            (open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset': DATASET}, "qald_id_big_data_test.json")))
    else:
        # Load the main data
        id_data = json.load(
            open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset': DATASET}, ID_BIG_DATA_FILENAME)))

        # Split it.
        id_data_test = test_split(id_data)
        id_data_train = train_split(id_data)

    if DENIS:
        test_pred = json.load(open(DENIS_FILE_LOCATION))
        for i in range(len(test_pred)):
            test_pred[i]['best_chain_id'] = pred_to_rel(test_pred[i]['best_chain'], relations)
            path_id = []
            for path in test_pred[i]['best_chain_id']:
                try:
                    path_id += [embeddings_interface.SPECIAL_CHARACTERS.index(path)]
                except ValueError:
                    path_id += relations[int(path)][3].tolist()

            for p in range(len(path_id)):
                try:
                    path_id[p] = vocab[path_id[p]]
                except KeyError:
                    print "key error"
                    path_id[p] = vocab[0]
            test_pred[i]['vocab_path_id'] = np.asarray(path_id)
        denis_pred = test_pred

    # Load all model
    with K.tf.device('/gpu:' + GPU):
        metric = rank_precision_metric(10)
        model_corechain = load_model(CORECHAIN_MODEL_DIR, {'custom_loss' :custom_loss, 'metric' :metric})
        model_rdf_type_check = load_model(RDFCHECK_MODEL_DIR, {'custom_loss' :custom_loss, 'metric' :metric})
        model_rdf_type_existence = load_model(RDFEXISTENCE_MODEL_DIR)
        model_question_intent = load_model(INTENT_MODEL_DIR)

    if DEBUG: print("the length of test data is ", len(id_data_test))
    if DEBUG: print("the dataset loaded is ", DATASET)


    for cnm in range(0 ,len(id_data_test)):
        log = {}
        data = id_data_test[cnm]
        log['id_data_text'] = data

        print counter
        counter = counter + 1
        if counter == 33 6 -1:
            continue
        '''
            Book keeping
        '''
        try:
            query_graph = {}
            rdf_type = True
            no_positive_path = False
            i_flag, c_flag, r_flag, rt_flag = False, False, False, False


            '''
                no_positive_path :- if the path had -1
            '''

            question, positive_path, negative_paths, no_positive_path = construct_paths(data, qald=True
                                                                                        ,relations=relations)

            '''
                if the dataset is LC-QUAD and data['pop'] is false then the positive path has been forcefully inserted and needs to be removed.
            '''
            if DATASET == 'lcquad':
                try:
                    if data['pop'] == False:
                        no_positive_path = True
                except KeyError:
                    pass

            if not no_positive_path:
                '''
                    If the positive path exists in the dataset.
                    This is use to prune candidate space using word2vec.

                    CANDIDATES_SPACE decides the number of candidates to keep 
                        If NONE :-  No pruning
                                - It is switched off permanently as it does not increase the performance.
                '''
                pp = [positive_path.tolist()]
                nps = [n.tolist() for n in negative_paths]
                paths = pp + nps
                if CANDIDATES_SPACE:
                    index = prune_candidate_space(question, paths, CANDIDATES_SPACE)
                    if index[-1] == 0:
                        word_vector_accuracy_counter = word_vector_accuracy_counter + 1
                else:
                    index = prune_candidate_space(question, paths, len(paths))
                paths = [paths[i] for i in index]
            else:
                nps = [n.tolist() for n in negative_paths]
                # pp = [positive_path.tolist()]
                paths =  nps
                if CANDIDATES_SPACE:
                    index = prune_candidate_space(question, paths, CANDIDATES_SPACE)
                else:
                    index = prune_candidate_space(question, paths, len(paths))
                paths = [paths[i] for i in index]

            '''
                Converting paths to numpy array
            '''
            for i in range(len(paths)):
                paths[i] = np.asarray(paths[i])
            paths = np.asarray(paths)


            '''
                Get the best path from the model
            '''
            previous_core_accuracy = core_chain_accuracy_counter
            if no_positive_path and DENIS:
                question, paths, positive_path, negative_paths, core_chain_accuracy_counter, best_path, mrr = core_chain_accuracy \
                    (question ,paths ,positive_path ,negative_paths ,core_chain_accuracy_counter ,model_corechain)
                if mrr != 0:
                    MRR.append(1. 0 /mrr)
                else:
                    MRR.append(0.0)

            elif DENIS:
                print "@denis"
                best_path = denis_pred[cnm]['vocab_path_id']
                if np.array_equal(positive_path ,best_path):
                    core_chain_accuracy_counter = core_chain_accuracy_counter + 1
            else:
                print "debug"
                question, paths, positive_path, negative_paths, core_chain_accuracy_counter, best_path, mrr = core_chain_accuracy(
                    question, paths, positive_path, negative_paths, core_chain_accuracy_counter, model_corechain
                    ,no_positive_path)
                print "the mmr is ", str(mrr)
                if mrr != 0:
                    MRR.append(1. 0 /mrr)
                else:
                    MRR.append(0.0)
            print "mrr is ", (sum(MRR) * 1.0 / len(MRR))
            query_graph['best_path'] = best_path
            if previous_core_accuracy + 1 == core_chain_accuracy_counter:
                c_flag = True
            padded_question = pad_question(question)

            # Predicting the intent of the question
            intent = question_intent(model_question_intent ,padded_question)
            query_graph['intent'] = intent

            if intent == ground_truth_intent(data):
                intent_accuracy_counter = intent_accuracy_counter + 1
                i_flag = True

            if query_graph['intent'] == 'ask':
                '''
                    if the intent is ask; then if there exists any path i.e either in negative or in positive path then 
                    the evaluation is true. 
                '''

                '''
                    finding the core chain accuracy for book keeping.
                '''


                if DATASET == 'qald':
                    if ground_truth_intent(data) == 'ask':
                        if data['pop'] == True or len(negative_paths) != 0:
                            if data['unparsed-qald-data']['answers'][0]['boolean']:
                                results.append([1.0, 1.0, 1.0])
                                avg.append(1.0)
                                print(sum(avg) * 1.0 / len(avg))
                                if i_flag and c_flag:
                                    query_graph_accuracy_counter = query_graph_accuracy_counter + 1

                            else:
                                results.append([0.0, 0.0, 0.0])
                                avg.append(0.0)
                                print(sum(avg) * 1.0 / len(avg))
                        else:
                            if data['unparsed-qald-data']['answers'][0]['boolean']:
                                results.append([0.0, 0.0, 0.0])
                                avg.append(0.0)
                                print(sum(avg) * 1.0 / len(avg))
                            else:
                                results.append([1.0, 1.0, 1.0])
                                avg.append(1.0)
                                print(sum(avg) * 1.0 / len(avg))
                                if i_flag and c_flag:
                                    query_graph_accuracy_counter = query_graph_accuracy_counter + 1
                    # raw_input()
                    i_flag, c_flag ,r_flag ,rt_flag = False ,False ,False ,False
                    continue


            if no_positive_path and len(negative_paths) == 0:
                '''
                    No candidate for positive path and negative path
                '''
                results.append([0.0, 0.0, 0.0])
                avg.append(0.0)
                print(sum(avg) * 1.0 / len(avg))
                i_flag, c_flag, r_flag, rt_flag = False, False, False, False
                log['result'] = [0.0 ,0.0 ,0.0]
                log['query_graph'] = query_graph
                Logging['main_log'].append(log)
                continue

            # Predicting the rdf-constraint
            rdf_constraint = rdf_constraint_check(padded_question ,model_rdf_type_existence)
            query_graph['rdf_constraint'] = False if rdf_constraint == 2 else True


            query_graph['rdf_constraint_type'] = ['x', 'uri', 'none'][rdf_constraint]
            query_graph['entities'] = data['parsed-data']['entity']
            ground_truth_rdf_intent, ground_truth_rdf_pred = ground_truth_rdf(data)
            if ground_truth_rdf_intent == query_graph['rdf_constraint_type']:
                r_flag = True


            if rdf_constraint != 2:

                rdf_candidates = rdf_type_candidates(data, best_path, vocab, relations, reverse_vocab, only_x=rdf_constraint == 0,
                                                     core_chain=True)
                '''
                    Predicting the rdf type constraints for the best core chain.
                '''
                if rdf_candidates:
                    output = rank_precision_runtime(model_rdf_type_check, question, rdf_candidates[0],
                                                    rdf_candidates, 180, MAX_SEQ_LENGTH ,rdf=True)
                    # rdf_best_path = convert_path_to_text(rdf_candidates[np.argmax(output[1:])])
                    query_graph['rdf_best_path'] = rdf_candidates[np.argmax(output[1:])]
                else:
                    query_graph['rdf_best_path'] = []

            type_pred = query_graph['intent']
            if query_graph['intent'] != ground_truth_intent(data):
                f ,p ,r = 0 ,0 ,0
            else:
                sparql = query_graph_to_sparql(query_graph)
                query_graph['sparql'] = sparql
                f, p, r = evaluate(sparql, data['parsed-data']['sparql_query'], type_pred, ground_truth_intent(data))
            results.append([f, p, r])
            avg.append(f)
            print(sum(avg) * 1.0 / len(avg))
            print "corechain accuracy is ", str(core_chain_accuracy_counter)
            # print sparql
            # print data['parsed-data']['sparql_query']
            if i_flag and c_flag and r_flag and rt_flag :
                query_graph_accuracy_counter = query_graph_accuracy_counter + 1
            i_flag ,c_flag ,r_flag ,rt_flag = False ,False ,False ,False
            log['result'] = [f ,p ,r]
            log['query_graph'] = query_graph
            Logging['main_log'].append(log)
        except UnboundLocalError:
            print traceback.print_exc()
            results.append([0.0, 0.0, 0.0])
            avg.append(0)
            print(sum(avg) * 1.0 / len(avg))
            log['result'] = results
            log['query_graph'] = query_graph
            Logging['main_log'].append(log)
            continue
    print"f1 is " ,str((sum(avg) * 1.0 / len(avg)))
    print "corechaibn accuracy is ", core_chain_accuracy_counter
    print "the final MRR is ", (sum(MRR) * 1.0 / len(MRR))
    print "the length of MRR is ", str(len(MRR))
    precision = [p[1] for p in results]
    recall = [r[2] for r in results]
    print "precision is ", str((sum(precision) * 1.0 / len(avg)))
    print "recall is ", str((sum(recall) * 1.0 / len(avg)))
    print CORECHAIN_MODEL_DIR