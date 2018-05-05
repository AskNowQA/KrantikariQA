# Shared Feature Extraction Layer
from __future__ import absolute_import
import os
import pickle
import sys
import json
import math
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import keras.backend.tensorflow_backend as K
from keras.layers.core import Layer  
from keras import initializers, regularizers, constraints
from keras.models import Model, Sequential
from keras.layers import Input, Layer, Lambda
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.layers import Activation, RepeatVector, Reshape, Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate, dot, subtract, maximum, multiply
from keras.layers import merge
from keras.activations import softmax
from keras import optimizers, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
from keras.callbacks import Callback
from keras.layers import InputSpec, Layer, Input, Dense, merge
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers import Bidirectional, GRU, LSTM
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import ELU
from keras.models import Sequential, Model, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Merge
from sklearn.utils import shuffle
import network as n
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils


# Some Macros
DEBUG = True
DATA_DIR = './data/training/pairwise'
RESOURCE_DIR = './resources_v8'
EPOCHS = 300
BATCH_SIZE = 180 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
NEGATIVE_SAMPLES = 270
OPTIMIZER = optimizers.Adam(LEARNING_RATE)


def load_data(file, max_sequence_length, relations):
    """
        Prepares the training data to be **directly** fed into the model.

    :param file:
    :param max_sequence_length:
    :param relations:
    :return:
    """
    glove_embeddings = n.get_glove_embeddings()

    try:

        # @TODO THIS BLOCK
        # with open(os.path.join(RESOURCE_DIR, file + ".mapped.npz")) as data, open(os.path.join(RESOURCE_DIR, file + ".vocab.pickle")) as idx:
        #     dataset = np.load(data)
        #     # dataset = dataset[:10]
        #     questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
        #     vocab = pickle.load(idx)
        #     vectors = glove_embeddings[vocab.keys()]
        #     return vectors, questions, pos_paths, neg_paths
        raise EOFError

    except (EOFError,IOError) as e:
        with open(os.path.join(RESOURCE_DIR, file)) as fp:
            dataset = pickle.load(fp)
            # dataset = dataset[:10]

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
                np.random.shuffle(unpadded_neg_path)
                unpadded_neg_path = pad_sequences(unpadded_neg_path, maxlen=max_sequence_length, padding='post')
                try:
                    neg_path = np.random.choice(unpadded_neg_path,200)
                except ValueError:
                    if len(unpadded_neg_path) == 0:
                        neg_path = neg_paths[-1]
                        print "Using previous question's paths for this since no neg paths for this question."
                    else:
                        index = np.random.randint(0, len(unpadded_neg_path), 200)
                        unpadded_neg_path = np.array(unpadded_neg_path)
                        neg_path = unpadded_neg_path[index]

                neg_paths.append(neg_path)


            # Convert things to nparrays
            questions = np.asarray(questions, dtype=np.int64)


            # questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
            neg_paths = np.asarray(neg_paths)



            '''
                Take care of vocabulary.
                Note: if vocabulary changed, all the models are rendered useless.
            '''

            try:
                vocab = pickle.load(open('resources_v8/id_big_data.json.vocab.pickle'))
            except (IOError, EOFError) as e:
                vocab = {}

            if DEBUG:
                print questions.shape
                print pos_paths.shape
                print neg_paths.shape

            all = np.concatenate([questions, pos_paths, neg_paths.reshape(neg_paths.shape[0]*neg_paths.shape[1],neg_paths.shape[2])], axis=0)
            uniques = np.unique(all)


            # ###########################################################
            # Map to new ID space all those which are not a part of vocab
            # ###########################################################

            index = len(vocab)

            for key in uniques:
                try:
                    temp = vocab[key]
                except KeyError:
                # if key not in vocab.keys():
                    vocab[key] = index
                    index += 1

            # Map everything
            for i in range(len(questions)):
                questions[i] = np.asarray([vocab[key] for key in questions[i]])
                pos_paths[i] = np.asarray([vocab[key] for key in pos_paths[i]])

                for j in range(len(neg_paths[i])):
                    neg_paths[i][j] = np.asarray([vocab[key] for key in neg_paths[i][j]])

            # Create slimmer, better, faster, vectors file.
            vectors = glove_embeddings[uniques]

            with open(os.path.join(RESOURCE_DIR, file + ".mapped.npz"), "w") as data, open(os.path.join(RESOURCE_DIR, file + ".vocab.pickle"), "w") as idx:
                np.savez(data, questions, pos_paths, neg_paths)
                pickle.dump(vocab,idx)

            return vectors, questions, pos_paths, neg_paths




if __name__ == "__main__":
    gpu = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    max_length = 25
    relations = n.load_relation('resources_v8/relations.pickle')
    dbp = db_interface.DBPedia(_verbose=True, caching=False)
    n.NEGATIVE_SAMPLES = 200
    n.BATCH_SIZE = 300
    vectors, questions, pos_paths, neg_paths = load_data("id_big_data.json", max_length, relations)
    print questions.shape
    n.main_bidirectional(gpu, vectors, questions, pos_paths, neg_paths,10, 200)
