from __future__ import absolute_import
import os
import pickle
import sys
import json
import numpy as np
import warnings
import keras.backend.tensorflow_backend as K
from keras import optimizers, metrics
from keras.layers import InputSpec, Layer, Input, Dense, merge
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers import Bidirectional, GRU, LSTM
from keras.models import Sequential, Model, model_from_json

import network as n
from utils import embeddings_interface
from utils import natural_language_utilities as nlutils

# Todos
# @TODO: The model doesn't take the embedding vectors as input.
# @TODO: Maybe put in negative sampling
# @TODO: Reuse code from network.py

# Setting a seed to clamp the stochastic nature of the code
np.random.seed(42)

# Some Macros
DEBUG = True
MAX_SEQ_LENGTH = 25
RAW_DATASET_LOC = './resources_v8/id_big_data.json'
DATA_DIR = './data/training/type-existence'

# Model Macros
EPOCHS = 300
BATCH_SIZE = 300 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
NEGATIVE_SAMPLES = 200
LOSS = 'categorical_crossentropy'
EMBEDDING_DIM = 300
OPTIMIZER = optimizers.Adam(LEARNING_RATE)

# Global variables
vocab = None

"""
    Training data is going to be
        X: a list of ID
        Y: count/list/ask

    Get X:
        - parse Iid-big-data file to get the question
        - get the final continous vocab dict to convert IDs
    Get Y:
        - parse their sparql to compute Y labels
"""


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def get_x(_datum):
    return np.asarray(embeddings_interface.vocabularize(nlutils.tokenize(_datum['parsed-data']['corrected_question'])),
                      dtype=np.int64)


def get_y(_datum):
    """
        Legend: 001: no
                010: uri
                100: x
    """

    # Check for ask
    if u'?uri' in _datum['parsed-data']['constraints'].keys():
        return np.asarray([0, 1, 0])

    if u'?x' in _datum['parsed-data']['constraints'].keys():
        return np.asarray([1, 0, 0])

    return np.asarray([0, 0, 1])


def create_dataset():
    """
        Open file
        Call getX, getY on every datapoint

    :return: two lists of dataset (train+test)
    """

    dataset = json.load(open(RAW_DATASET_LOC))

    X = np.zeros((len(dataset), MAX_SEQ_LENGTH), dtype=np.int64)
    Y = np.zeros((len(dataset), 3), dtype=np.int64)

    for i in range(len(dataset)):
        data = dataset[i]

        # Call fns to parse it
        x, y = get_x(data), get_y(data)

        # Append ze data into their lists
        X[i, :min(x.shape[0], MAX_SEQ_LENGTH)] = x[:min(x.shape[0], MAX_SEQ_LENGTH)]
        Y[i] = y

    # Convert to new (continous IDs)
    vectors, X, Y = convert_new_ids(X, Y)

    # Shuffle
    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    Y = Y[s]

    # Split
    train_X, test_X = X[:int(X.shape[0]*0.8)], X[int(X.shape[0]*0.8):]
    train_Y, test_Y = Y[:int(Y.shape[0]*0.8)], Y[int(Y.shape[0]*0.8):]

    # # Save
    # np.save(open(os.path.join(MODEL_DIR, 'trainX.npy'), 'w+'), train_X)
    # np.save(open(os.path.join(MODEL_DIR, 'trainY.npy'), 'w+'), train_Y)
    # np.save(open(os.path.join(MODEL_DIR, 'testX.npy'), 'w+'), test_X)
    # np.save(open(os.path.join(MODEL_DIR, 'testY.npy'), 'w+'), test_Y)

    return vectors, train_X, test_X, train_Y, test_Y


def convert_new_ids(X, Y):
    """
        If not vocabulary, pull from disk, and add stuff to it.
    :param X: numpy mat n, 25
    :param Y: numpy mat n, 3
    :return: reduced embedding mat, X (converted), Y
    """
    global vocab

    # Collect the embedding matrix.
    glove_embeddings = get_glove_embeddings()

    # See if we've already loaded the new vocab.
    if not vocab:
        try:
            vocab = pickle.load(open('resources_v8/id_big_data.json.vocab.pickle'))
        except (IOError, EOFError) as e:
            if DEBUG:
                warnings.warn("Did not find the vocabulary.")
            vocab = {}

    # Collect uniques from X
    uniques = np.unique(X)

    # Update vocab in case of unseen questions
    index = len(vocab)

    for key in uniques:
        try:
            temp = vocab[key]
        except KeyError:
            # if key not in vocab.keys():
            vocab[key] = index
            index += 1

    # Map X
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            X[i][j] = vocab[X[i][j]]

    # Reduce Embedding Matrix
    vectors = glove_embeddings[uniques]

    # Return stuff
    return vectors, X, Y


def get_glove_embeddings():
    from utils.embeddings_interface import __check_prepared__
    __check_prepared__('glove')
    from utils.embeddings_interface import glove_embeddings
    return glove_embeddings


def rnn_model(embedding_layer, X_train, Y_train, max_seq_length):
    """
        The simplest model at hand to do this job.
        A Bidirectional LSTM, A dense and a dense softmax output layer.

    :param embedding_layer:
    :param X_train:
    :param Y_train:
    :param max_seq_length:
    :return:
    """
    sequence_input = Input(shape=(max_seq_length,))
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(128, dropout=0.5))(embedded_sequences)
    x = Dense(128, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss=LOSS,
                  optimizer=OPTIMIZER,
                  metrics=['acc'])
    model.summary()

    # Training time bois.
    model.fit(np.asarray(X_train), np.asarray(Y_train),
              epochs=30, batch_size=128)

    return model


def run(vectors, X_train, Y_train, gpu):
    """
        File which instantiates the model.
        Puts in the embedding layer and everything else.
        Calls evaluate and gets results

    :param vectors:
    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    """
    with K.tf.device('/gpu:' + gpu):
        # Construct a common embedding layer
        embedding_layer = Embedding(vectors.shape[0],
                                    EMBEDDING_DIM,
                                    weights=[vectors],
                                    input_length=MAX_SEQ_LENGTH,
                                    trainable=False)

        # Train
        model = rnn_model(embedding_layer, X_train, Y_train, MAX_SEQ_LENGTH)

        return model


if __name__ == "__main__":

    gpu = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # n.NEGATIVE_SAMPLES = NEGATIVE_SAMPLES
    # n.BATCH_SIZE = BATCH_SIZE

    vectors, X_train, X_test, Y_train, Y_test = create_dataset()

    model = run(vectors, X_train, Y_train, gpu)

    # Predict
    Y_test_cap = model.predict(X_test)

    # Evaluate
    result = 0
    for i in range(len(Y_test_cap)):
        if np.argmax(Y_test_cap[i]) == np.argmax(Y_test[i]):
            result = result + 1
    print("rnn model results are ", result/float(len(Y_test_cap)))

    # Time to save model.
    n.DATA_DIR = DATA_DIR
    n.smart_save_model(model)