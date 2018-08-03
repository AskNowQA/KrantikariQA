"""
    The file is used to comb through all the data files and create a condensed vocabulary.
    Then, the vocabulary and the cropped word vectors file is to be stored somewhere.

    In case it isn't found there, just call prepare again.
"""

# Shared Feature Extraction Layer
from __future__ import absolute_import
import os
import pickle
import sys
import json
import math
import atexit
import warnings
import numpy as np
from sklearn.utils import shuffle


import keras.backend.tensorflow_backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import concatenate, dot
from keras import optimizers, metrics
from keras.utils import Sequence
from keras.callbacks import Callback
from keras.layers import InputSpec, Layer, Input, Dense, merge
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed, concatenate, Conv1D, MaxPooling1D, Embedding, Flatten
from keras.layers import Bidirectional, GRU, LSTM
from keras.models import Sequential, Model, model_from_json
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.regularizers import L1L2

try:
    from utils import embeddings_interface
except ImportError:
    import embeddings_interface

try:
    from utils import dbpedia_interface as db_interface
except ImportError:
    import dbpedia_interface as db_interface

try:
    from utils import natural_language_utilities as nlutils
except ImportError:
    import natural_language_utilities as nlutils


# Macros
MAX_SEQUENCE_LENGTH = 25
DEBUG = True

# Data locations
MODEL_DIR = './data/models/core_chain/lcquad/'
MODEL_SPECIFIC_DATA_DIR = './data/data/core_chain_pairwise/%d(dataset)s/'
COMMON_DATA_DIR = './data/data/common/'
DATASET_SPECIFIC_DATA_DIR = './data/data/%(dataset)s/'

vocab = None
vectors = None
dbp = db_interface.DBPedia(_verbose=True, caching=False)


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def _load_relations_():
    relations = pickle.load(open(COMMON_DATA_DIR+'/relations.pickle'))
    inverse_relations = {}
    for key in relations:
        value = relations[key]
        new_key = value[0]
        value[0] = key
        inverse_relations[new_key] = value

    return inverse_relations


def _get_glove_embeddings_():
    try:
        from utils.embeddings_interface import __check_prepared__
    except ImportError:
        from embeddings_interface import __check_prepared__
    __check_prepared__('glove')

    try:
        from utils.embeddings_interface import glove_embeddings
    except ImportError:
        from embeddings_interface import glove_embeddings
    return glove_embeddings


def _prepare_():
    """
        First, go through the questions, pos paths, neg paths in LCQUAD, then in QALD and make everything.
    """

    global dbp

    try:
        vocab = pickle.load(open(os.path.join(COMMON_DATA_DIR, "vocab.pickle")))
        vectors = np.load(open(os.path.join(COMMON_DATA_DIR, "vectors.npy")))
        print("length of vocab file is before calling prepare i.e current vocab file is ", len(vocab))
        print("length of vector file is before calling prepare i.e current vector file is ", len(vocab))
    except (IOError, EOFError) as e:
        if DEBUG:
            warnings.warn("Did not find the vocabulary.")
        vocab = {}
        vectors = None

    file = "id_big_data.json"
    max_sequence_length = MAX_SEQUENCE_LENGTH
    relations = _load_relations_()
    glove_embeddings = _get_glove_embeddings_()

    # LCQuAD time
    with open(os.path.join(DATASET_SPECIFIC_DATA_DIR % {'dataset':'lcquad'}, file)) as fp:
        dataset = json.load(fp)
        # dataset = dataset[:10]
        questions = [i['uri']['question-id'] for i in dataset]
        questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
        pos_paths = []
        for i in dataset:
            path_id = i['parsed-data']['path_id']
            positive_path = []
            for p in path_id:
                positive_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p[0])]
                positive_path += relations[int(p[1:])][3].tolist()
            pos_paths.append(positive_path)

        neg_paths = []
        for i in range(0, len(dataset)):
            datum = dataset[i]
            negative_paths_id = datum['uri']['hop-2-properties'] + datum['uri']['hop-1-properties']
            np.random.shuffle(negative_paths_id)
            negative_paths = []
            for neg_path in negative_paths_id:
                negative_path = []
                for p in neg_path:
                    try:
                        negative_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p)]
                    except ValueError:
                        negative_path += relations[int(p)][3].tolist()
                negative_paths.append(negative_path)
            # negative_paths = remove_positive_path(pos_paths[i], negative_paths)
            # try:
            #     negative_paths = np.random.choice(negative_paths, 1000)
            # except ValueError:
            #     if len(negative_paths) == 0:
            #         negative_paths = neg_paths[-1]
            #         print("Using previous question's paths for this since no neg paths for this question.")
            #     else:
            #         index = np.random.randint(0, len(negative_paths), 1000)
            #         negative_paths = np.array(negative_paths)
            #         negative_paths = negative_paths[index]
            neg_paths.append(negative_paths)

        for i in range(0, len(neg_paths)):
            neg_paths[i] = pad_sequences(neg_paths[i], maxlen=max_sequence_length, padding='post')

        neg_paths = np.asarray(neg_paths)
        pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
        all_lcquad_nonneg = np.unique(np.concatenate([questions, pos_paths], axis=0))
        all_lcquad_neg = neg_paths[0].flatten()
        for neg_path in neg_paths:
            all_lcquad_neg = np.concatenate((all_lcquad_neg, neg_path.flatten()))
            all_lcquad_neg = np.unique(all_lcquad_neg)
        all_lcquad_neg = np.unique(all_lcquad_neg)
        # all_lcquad_neg = np.unique(np.asarray([ x.flatten() for x in neg_paths]))
        # for neg_path in neg_paths:
        #     if not all_lcquad_neg:
        #         all_lcquad_neg = np.unique()
        # all_lcquad_neg = np.unique(neg_paths)
        uniques_lcquad = np.unique((np.concatenate((all_lcquad_neg, all_lcquad_nonneg))))


        # LCQuAD with RDF-Type shit.
        # Empty arrays
        questions = []
        pos_paths = []
        neg_paths = []

        for i in range(len(dataset)):

            datum = dataset[i]

            '''
                Extracting and padding the positive paths.
            '''
            if '?x' in datum['parsed-data']['constraints'].keys():
                pos_path = "x + " + dbp.get_label(datum['parsed-data']['constraints']['?x'])
            elif '?uri' in datum['parsed-data']['constraints'].keys():
                pos_path = "uri + " + dbp.get_label(datum['parsed-data']['constraints']['?uri'])
            else:
                continue
            pos_path = embeddings_interface.vocabularize(nlutils.tokenize(pos_path))
            pos_paths.append(pos_path)

            # Question
            question = np.zeros((max_sequence_length), dtype=np.int64)
            unpadded_question = np.asarray(datum['uri']['question-id'])
            question[:min(len(unpadded_question), max_sequence_length)] = unpadded_question
            questions.append(question)

            # Negative Path
            unpadded_neg_path = datum["rdf-type-constraints"]
            # unpadded_neg_path = n.remove_positive_path(pos_path, unpadded_neg_path)
            np.random.shuffle(unpadded_neg_path)
            unpadded_neg_path = pad_sequences(unpadded_neg_path, maxlen=max_sequence_length, padding='post')

            '''
                Remove positive path from negative paths.
            '''

            # try:
            #     neg_path = np.random.choice(unpadded_neg_path, 200)
            # except ValueError:
            #     if len(unpadded_neg_path) == 0:
            #         neg_path = neg_paths[-1]
            #         print("Using previous question's paths for this since no neg paths for this question.")
            #     else:
            #         index = np.random.randint(0, len(unpadded_neg_path), 200)
            #         unpadded_neg_path = np.array(unpadded_neg_path)
            #         neg_path = unpadded_neg_path[index]

            neg_paths.append(unpadded_neg_path)

        # Convert things to nparrays
        questions = np.asarray(questions, dtype=np.int64)

        # questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
        pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
        neg_paths = np.asarray(neg_paths)
        all_lcquad_rdf_nonneg = np.unique(np.concatenate([questions, pos_paths], axis=0))
        # all_lcquad_rdf_neg = np.unique(neg_paths)
        # all_lcquad_rdf_neg = np.unique(np.asarray([x.flatten() for x in neg_paths]))
        all_lcquad_rdf_neg = neg_paths[0].flatten()
        for neg_path in neg_paths:
            all_lcquad_rdf_neg = np.concatenate((all_lcquad_rdf_neg, neg_path.flatten()))
            all_lcquad_rdf_neg = np.unique(all_lcquad_rdf_neg)
        all_lcquad_rdf_neg = np.unique(all_lcquad_rdf_neg)
        uniques_lcquad_rdf = np.unique(np.concatenate((all_lcquad_rdf_neg, all_lcquad_rdf_nonneg)))

    # QALD Time
    qald_train = json.load(open(DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'} + "/qald_id_big_data_train.json"))
    qald_test = json.load(open(DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'} + "/qald_id_big_data_test.json"))

    dataset = qald_train + qald_test

    pos_paths = []
    paths_to_ignore = []
    for i in dataset:
        path_id = i['parsed-data']['path_id']
        positive_path = []
        try:
            for p in path_id:
                positive_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p[0])]
                positive_path += relations[int(p[1:])][3].tolist()
        except (TypeError, ValueError) as e:
            paths_to_ignore.append(i)
            continue
        pos_paths.append(positive_path)

    questions = [i['uri']['question-id'] for i in dataset]
    questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')

    neg_paths = []
    for i in range(0, len(dataset)):
        # if i in paths_to_ignore:
        #     continue
        datum = dataset[i]
        negative_paths_id = datum['uri']['hop-2-properties'] + datum['uri']['hop-1-properties']
        np.random.shuffle(negative_paths_id)
        negative_paths = []
        for neg_path in negative_paths_id:
            negative_path = []
            for p in neg_path:
                try:
                    negative_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p)]
                except ValueError:
                    negative_path += relations[int(p)][3].tolist()
            negative_paths.append(negative_path)
        # try:
        #     negative_paths = np.random.choice(negative_paths, 1000)
        # except ValueError:
        #     if len(negative_paths) == 0:
        #         negative_paths = neg_paths[-1]
        #         print("Using previous question's paths for this since no neg paths for this question.")
        #     else:
        #         index = np.random.randint(0, len(negative_paths), 1000)
        #         negative_paths = np.array(negative_paths)
        #         negative_paths = negative_paths[index]
        neg_paths.append(negative_paths)

    for i in range(0, len(neg_paths)):
        neg_paths[i] = pad_sequences(neg_paths[i], maxlen=max_sequence_length, padding='post')
    neg_paths = np.asarray(neg_paths)
    pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
    all_qald_nonneg = np.unique(np.concatenate([questions, pos_paths], axis=0))
    # all_qald_neg = np.unique(neg_paths)
    # all_qald_neg = np.unique(np.asarray([x.flatten() for x in neg_paths]))
    all_qald_neg = neg_paths[0].flatten()
    for neg_path in neg_paths:
        all_qald_neg = np.concatenate((all_qald_neg , neg_path.flatten()))
        all_qald_neg = np.unique(all_qald_neg )
    all_qald_neg = np.unique(all_qald_neg )
    uniques_qald = np.unique(np.concatenate((all_qald_nonneg, all_qald_neg)))

    # QALD rdftype
    # Empty arrays
    questions = []
    pos_paths = []
    neg_paths = []

    for i in range(len(dataset)):

        datum = dataset[i]

        '''
            Extracting and padding the positive paths.
        '''
        if '?x' in datum['parsed-data']['constraints'].keys():
            pos_path = "x + " + dbp.get_label(datum['parsed-data']['constraints']['?x'])
        elif '?uri' in datum['parsed-data']['constraints'].keys():
            pos_path = "uri + " + dbp.get_label(datum['parsed-data']['constraints']['?uri'])
        else:
            continue
        pos_path = embeddings_interface.vocabularize(nlutils.tokenize(pos_path))
        pos_paths.append(pos_path)

        # Question
        question = np.zeros((max_sequence_length), dtype=np.int64)
        unpadded_question = np.asarray(datum['uri']['question-id'])
        question[:min(len(unpadded_question), max_sequence_length)] = unpadded_question
        questions.append(question)

        # Negative Path
        unpadded_neg_path = datum["rdf-type-constraints"]
        # unpadded_neg_path = n.remove_positive_path(pos_path, unpadded_neg_path)
        np.random.shuffle(unpadded_neg_path)
        unpadded_neg_path = pad_sequences(unpadded_neg_path, maxlen=max_sequence_length, padding='post')

        '''
            Remove positive path from negative paths.
        '''

        # try:
        #     neg_path = np.random.choice(unpadded_neg_path, 200)
        # except ValueError:
        #     if len(unpadded_neg_path) == 0:
        #         neg_path = neg_paths[-1]
        #         print("Using previous question's paths for this since no neg paths for this question.")
        #     else:
        #         index = np.random.randint(0, len(unpadded_neg_path), 200)
        #         unpadded_neg_path = np.array(unpadded_neg_path)
        #         neg_path = unpadded_neg_path[index]

        neg_paths.append(unpadded_neg_path)

    # Convert things to nparrays
    questions = np.asarray(questions, dtype=np.int64)

    # questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
    pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
    neg_paths = np.asarray(neg_paths)
    all_qald_rdf_nonneg = np.unique(np.concatenate((questions, pos_paths), axis=0))
    # all_qald_rdf_neg = np.unique(neg_paths)
    # all_qald_rdf_neg = np.unique(np.asarray([x.flatten() for x in neg_paths]))
    all_qald_rdf_neg = neg_paths[0].flatten()
    for neg_path in neg_paths:
        all_qald_rdf_neg = np.concatenate((all_qald_rdf_neg, neg_path.flatten()))
        all_qald_rdf_neg = np.unique(all_qald_rdf_neg)
    all_qald_rdf_neg = np.unique(all_qald_rdf_neg)
    uniques_qald_rdf = np.unique(np.concatenate((all_qald_rdf_neg, all_qald_rdf_nonneg)))

    # # Now to build the vocab
    # uniques_lcquad = np.unique(all_lcquad)
    # uniques_lcquad_rdf = np.unique(all_lcquad_rdf)
    # uniques_qald = np.unique(all_qald)
    # uniques_qald_rdf = np.unique(all_qald_rdf)

    for i in range(len(uniques_lcquad)):
        try:
            temp = vocab[uniques_lcquad[i]]
        except KeyError:
            vocab[uniques_lcquad[i]] = len(vocab.keys())

    for i in range(len(uniques_lcquad_rdf)):
        try:
            temp = vocab[uniques_lcquad_rdf[i]]
        except KeyError:
            vocab[uniques_lcquad_rdf[i]] = len(vocab.keys())

    for i in range(len(uniques_qald)):
        try:
            temp = vocab[uniques_qald[i]]
        except KeyError:
            vocab[uniques_qald[i]] = len(vocab.keys())

    for i in range(len(uniques_qald_rdf)):
        try:
            temp = vocab[uniques_qald_rdf[i]]
        except KeyError:
            vocab[uniques_qald_rdf[i]] = len(vocab.keys())

    vectors = glove_embeddings[sorted(vocab.keys())]

    # Save these sons of bitches.
    if DEBUG: print("Vectors and Vocab prepared. Now, we gotsta save 'em")
    print("new length of vocab file is before calling prepare i.e new current vocab file is ", len(vocab))
    print("new length of vector file is before calling prepare i.e new current vector file is ", len(vocab))
    pickle.dump(vocab, open(os.path.join(COMMON_DATA_DIR, "vocab.pickle"), 'w+'))
    np.save(os.path.join(COMMON_DATA_DIR, "vectors"), vectors)

    return vocab, vectors


def load():
    """
        To be called by different files to load the vectors and vocab.
    """
    global vocab, vectors

    try:
        vocab = pickle.load(open(os.path.join(COMMON_DATA_DIR, "vocab.pickle")))
        vectors = np.load(open(os.path.join(COMMON_DATA_DIR, "vectors.npy" )))
    except (IOError, EOFError) as e:
        if DEBUG:
            warnings.warn("Did not find the vocabulary.")

        vocab, vectors = _prepare_()

    return vocab, vectors


# def convert(data, singular=False):
#     """
#
#         Assumes either a single int or a list (1D)
#
#     :param data: int/1D list of IDs
#     :param singular: boolean to specify if we have a single int or a list
#     :return: 1D numpy array or an int (depends on singluar flag)
#     """
#     global vectors, vocab
#
#     try:
#         assert vocab
#         assert vectors
#     except AssertionError:
#         prepare()
#
#     if singular:
#         return vocab[data]
#
#     return np.asarray([vocab[key] for key in data])


if __name__ == "__main__":
    _prepare_()