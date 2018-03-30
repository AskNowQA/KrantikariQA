"""
    This file contains parsing and model code for predicting the rdf:type relation (whether or not we have an rdf:type
        and on which variable)
"""
import os
import re
import json
import pickle
import numpy as np
from pprint import pprint

from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Embedding, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D, Reshape, Flatten, Dropout, LSTM, Bidirectional

from utils import natural_language_utilities as nlutils
from utils import embeddings_interface
from utils.embeddings_interface import glove_embeddings as glove

np.random.seed(42)

# Some Macros
DATASET_DIR = './data/typeprediction/dataset/'
MODEL_DIR = './data/typeprediction/model/'
RAW_DATASET_LOC = './resources/data_set.json'
EMBEDDING_DIM = 300
DEBUG = True

# Some regexes
uri_type_re = "\?[uri]*\s*(rdf:type|<http:\/\/www.w3.org\/.*#type>)"
x_type_re = "\?[x]*\s*(rdf:type|<http:\/\/www.w3.org\/.*#type>)"


"""
    Training data is going to be
        X: a list of ID
        Y: x/uri/none

    Get X:
        - parse json file to find questions
        - convert them to ids using embeddings interface
    Get Y:
        - parse their sparql to compute Y labels
"""


def get_x(_datum):
    return embeddings_interface.vocabularize(nlutils.tokenize(_datum['corrected_question']))


def get_y(_datum):
    """
        Legend: 001: none
                010: uri
                100: x
    """

    # Check if the constraint's on URI
    uri_matcher = re.search(uri_type_re, _datum['sparql_query'])
    x_matcher = re.search(x_type_re, _datum['sparql_query'])

    if uri_matcher and x_matcher:
        if DEBUG:
            print("\n\nFOUND BOTH URI AND X CONSTRAINTS IN ")
            pprint(_datum)
            print("\n\n")
        return np.asarray([0, 0, 0])

    elif uri_matcher:
        return np.asarray([0, 1, 0])

    elif x_matcher:
        return np.asarray([1, 0, 0])

    else:
        return np.asarray([0, 0, 1])


def create_dataset():
    """
        Open file
        Call getX, getY on every datapoint

    :return: two lists of dataset (train+test)
    """
    max_len = 0
    X_list = []
    Y = np.zeros((5000, 3))

    dataset = json.load(open(RAW_DATASET_LOC))

    for i in range(len(dataset)):
        data = dataset[i]

        # Call fns to parse it
        x, y = get_x(data), get_y(data)

        # Append ze data into their lists
        X_list.append(x)
        Y[i] = y

        # Calculate the max length (for padding later)
        if len(x) > max_len: max_len = len(x)

    # Convert to numpy
    X_list = np.asarray(X_list)

    # Shuffle
    s = np.arange(X_list.shape[0])
    np.random.shuffle(s)
    X_list = X_list[s]
    Y = Y[s]

    # Pad
    X = np.zeros((5000, max_len))
    for i in range(X_list.shape[0]):
        X[i,: X_list[i].shape[0]] = X_list[i]
    # X[:X_list.shape[0], :X_list.shape[1]] = X_list

    # Split
    train_X, test_X = X[:int(X.shape[0]*0.8)], X[int(X.shape[0]*0.8):]
    train_Y, test_Y = Y[:int(Y.shape[0]*0.8)], Y[int(Y.shape[0]*0.8):]

    # # Save
    # np.save(open(os.path.join(MODEL_DIR, 'trainX.npy'), 'w+'), train_X)
    # np.save(open(os.path.join(MODEL_DIR, 'trainY.npy'), 'w+'), train_Y)
    # np.save(open(os.path.join(MODEL_DIR, 'testX.npy'), 'w+'), test_X)
    # np.save(open(os.path.join(MODEL_DIR, 'testY.npy'), 'w+'), test_Y)

    return train_X, train_Y, test_X, test_Y, max_len


def get_glove_embeddings():
    from utils.embeddings_interface import __check_prepared__
    __check_prepared__('glove')
    from utils.embeddings_interface import glove_embeddings
    return glove_embeddings

def reduce_embeddings_mat(train_X, test_X):

    glove_embeddings = get_glove_embeddings()
    id_mapper = [] # Index: new id; val: old id
    current_id = 0

    train_mapped, test_mapped = np.zeros(train_X.shape), np.zeros(test_X.shape)

    # First on train set
    for i in range(len(train_X)):
        question = train_X[i]
        for j in range(len(question)):
            token = question[j]

            try:
                train_mapped[i][j] = id_mapper.index(token)
            except ValueError:
                train_mapped[i][j] = current_id
                id_mapper.append(train_mapped[i][j])
                current_id += 1

    # Now on test set
    for i in range(len(test_X)):
        question = test_X[i]
        for j in range(len(question)):
            token = question[j]

            try:
                test_mapped[i][j] = id_mapper.index(token)
            except ValueError:
                test_mapped[i][j] = current_id
                id_mapper.append(test_mapped[i][j])
                current_id += 1

    # Calculate small embedding matrix
    smaller_embedding_matrix = np.zeros((len(id_mapper), 300))
    for i in range(len(id_mapper)):
        key = int(id_mapper[i])
        smaller_embedding_matrix[i] = glove_embeddings[key]

    return train_mapped, test_mapped, smaller_embedding_matrix, id_mapper


def cnn_model(embedding_layer, X_train, Y_train, max_seq_length):
    sequence_input = Input(shape=(max_seq_length,))
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', input_shape=(25, 300))(embedded_sequences)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(2)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    model.fit(np.asarray(X_train), np.asarray(Y_train),
              epochs=30, batch_size=128)
    return model


def rnn_model(embedding_layer, X_train, Y_train, max_seq_length):
    sequence_input = Input(shape=(max_seq_length,))
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(128, dropout=0.5))(embedded_sequences)
    x = Dense(128, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    model.fit(np.asarray(X_train), np.asarray(Y_train),
              epochs=30, batch_size=128)
    return model


def rnn_cnn_model(embedding_layer, X_train, Y_train, max_seq_length):
    sequence_input = Input(shape=(max_seq_length,))
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', input_shape=(25, 300))(embedded_sequences)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(2)(x)  # global max pooling
    # x = Flatten()(x)
    x = LSTM(128)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    model.fit(np.asarray(X_train), np.asarray(Y_train),
              epochs=30, batch_size=128)
    return model


if __name__ == "__main__":
    """
        The orchestrating function around here. I take care of everything. I am the changer.

            - check if we already have a dataset or not.
                - if not: we call create_dataset again

            - make the IDs continous; and make the same change in the matrix

    :return:
    """

    """
        Load/Create dataset
    """
    train_X, train_Y, test_X, test_Y, max_len = create_dataset()

    """
        Make IDs continous
    """
    train_X, test_X, embeddings_mat, vocab_dict = reduce_embeddings_mat(train_X, test_X)

    """
        Start the training part
    """
    # Construct a common embedding layer
    embedding_layer = Embedding(len(vocab_dict),
                                EMBEDDING_DIM,
                                weights=[embeddings_mat],
                                input_length=max_len,
                                trainable=False)

    # Train
    cnn_model = cnn_model(embedding_layer, train_X, train_Y, max_len)
    rnn_model = rnn_model(embedding_layer, train_X, train_Y, max_len)
    rnn_cnn_model = rnn_cnn_model(embedding_layer, train_X, train_Y, max_len)

    # Predict
    cnn_model_predict = cnn_model.predict(test_X)
    rnn_model_predict = rnn_model.predict(test_X)
    rnn_cnn_model_predict = rnn_cnn_model.predict(test_X)

    # Evaluate
    result = 0
    for i in xrange(len(cnn_model_predict)):
        if np.argmax(cnn_model_predict[i]) == np.argmax(test_Y[i]) or np.argmax(rnn_model_predict[i]) == np.argmax(
                test_Y[i]):
            result = result + 1

    print "combined results are ", result

    result = 0
    for i in xrange(len(cnn_model_predict)):
        if np.argmax(cnn_model_predict[i]) == np.argmax(test_Y[i]):
            result = result + 1
    print "cnn model results are ", result

    result = 0
    for i in xrange(len(rnn_model_predict)):
        if np.argmax(rnn_model_predict[i]) == np.argmax(test_Y[i]):
            result = result + 1
    print "rnn model results are ", result

    result = 0
    for i in xrange(len(rnn_cnn_model_predict)):
        if np.argmax(rnn_cnn_model_predict[i]) == np.argmax(test_Y[i]):
            result = result + 1
    print "rnn model results are ", result

    result = 0
    for i in xrange(len(cnn_model_predict)):
        if np.argmax(cnn_model_predict[i]) == np.argmax(test_Y[i]) or np.argmax(rnn_model_predict[i]) == np.argmax(
                test_Y[i]) \
                or np.argmax(rnn_cnn_model_predict[i]) == np.argmax(test_Y[i]):
            result = result + 1

    print "combined results are ", result