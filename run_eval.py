

from __future__ import absolute_import

import os

gpu = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

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


# Some Macros
DEBUG = True
DATA_DIR = './data/training/pairwise'
EPOCHS = 300
BATCH_SIZE = 880 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
NEGATIVE_SAMPLES = 1000
OPTIMIZER = optimizers.Adam(LEARNING_RATE)


def custom_loss(y_true, y_pred):
    '''
        max margin loss
    '''
    # y_pos = y_pred[0]
    # y_neg= y_pred[1]
    diff = y_pred[:,-1]
    # return K.sum(K.maximum(1.0 - diff, 0.))
    return K.sum(diff)

import torch as t


def rank_precision(model, test_questions, test_pos_paths, test_neg_paths, neg_paths_per_epoch=100, batch_size=1000):
    max_length = test_questions.shape[-1]
    questions = np.reshape(np.repeat(np.reshape(test_questions,
                                            (test_questions.shape[0], 1, test_questions.shape[1])),
                                 neg_paths_per_epoch+1, axis=1), (-1, max_length))
    pos_paths = np.reshape(test_pos_paths,
                                    (test_pos_paths.shape[0], 1, test_pos_paths.shape[1]))
    neg_paths = test_neg_paths[:, np.random.randint(0, NEGATIVE_SAMPLES, neg_paths_per_epoch), :]
    all_paths = np.reshape(np.concatenate([pos_paths, neg_paths], axis=1), (-1, max_length))

    outputs = model.predict([questions, all_paths, np.zeros_like(all_paths)], batch_size=batch_size)[:,0]
    outputs = np.reshape(outputs, (test_questions.shape[0], neg_paths_per_epoch+1))

    precision = float(len(np.where(np.argmax(outputs, axis=1)==0)[0]))/outputs.shape[0]
    return precision



def rank_precision_metric(neg_paths_per_epoch):
    def metric(y_true, y_pred):
        scores = y_pred[:, 0]
        scores = K.reshape(scores, (-1, neg_paths_per_epoch+1))
        hits = K.cast(K.shape(K.tf.where(K.tf.equal(K.tf.argmax(scores, axis=1),0)))[0], 'float32')
        precision = hits/K.cast(K.shape(scores)[0], 'float32')
        # precision = float(len(np.where(np.argmax(all_outputs, axis=1)==0)[0]))/all_outputs.shape[0]
        return precision
    return metric

def get_glove_embeddings():
    from utils.embeddings_interface import __check_prepared__
    __check_prepared__('glove')

    from utils.embeddings_interface import glove_embeddings
    return glove_embeddings

# class CustomLossHistory(Callback):
#     def __init__(self, loss, validation_set):
#         self.loss = loss
#         self.validation_data = validation_set # validation_set = x, y
#
#     def on_train_begin(self, logs={}):
#         self.losses = []
#
#     def on_batch_end(self, batch, logs={}):
#         current_loss_value = self.loss(self.validation_data[1],
#             self.model.predict(self.validation_data[0]))
#         print current_loss_value
#         self.losses.append(current_loss_value)
#         # You could also print it out here.


def cross_correlation(x):
    a, b = x
    tf = K.tf
    a_fft = tf.fft(tf.complex(a, 0.0))
    b_fft = tf.fft(tf.complex(b, 0.0))
    ifft = tf.ifft(tf.conj(a_fft) * b_fft)
    return tf.cast(tf.real(ifft), 'float32')

def load_data(file, max_sequence_length):
    # glove_embeddings = get_glove_embeddings()

    try:
        with open(os.path.join(DATA_DIR, file + ".mapped.npz")) as data, open(os.path.join(DATA_DIR, file + ".index.npy")) as idx:
            dataset = np.load(data)
            questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
            index = np.load(idx)
            # vectors = glove_embeddings[index]
            return None, questions, pos_paths, neg_paths
    except:
        with open(os.path.join(DATA_DIR, file)) as fp:
            dataset = pickle.load(fp)
            questions = [i[0] for i in dataset]
            questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
            pos_paths = [i[1] for i in dataset]
            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
            neg_paths = [i[2] for i in dataset]
            neg_paths = [path for paths in neg_paths for path in paths]
            neg_paths = pad_sequences(neg_paths, maxlen=max_sequence_length, padding='post')

            all = np.concatenate([questions, pos_paths, neg_paths], axis=0)
            mapped_all, index = pd.factorize(all.flatten(), sort=True)
            mapped_all = mapped_all.reshape((-1, max_sequence_length))
            vectors = glove_embeddings[index]

            questions, pos_paths, neg_paths = np.split(mapped_all, [questions.shape[0], questions.shape[0]*2])
            neg_paths = np.reshape(neg_paths, (len(questions), NEGATIVE_SAMPLES, max_sequence_length))

            with open(os.path.join(DATA_DIR, file + ".mapped.npz"), "w") as data, open(os.path.join(DATA_DIR, file + ".index.npy"), "w") as idx:
                np.savez(data, questions, pos_paths, neg_paths)
                np.save(idx, index)

            return vectors, questions, pos_paths, neg_paths



gpu = '1'


"""
    Data Time!
"""
# Pull the data up from disk
max_length = 50
vectors, questions, pos_paths, neg_paths = load_data("results_jan_12_full.pickle", max_length)
# pad_till = abs(pos_paths.shape[1] - questions.shape[1])
# pad = lambda x: np.pad(x, [(0,0), (0,pad_till), (0,0)], 'constant', constant_values=0.)
# if pos_paths.shape[1] < questions.shape[1]:
#     pos_paths = pad(pos_paths)
#     neg_paths = pad(neg_paths)
# else:
#     questions = pad(questions)

# Shuffle these matrices together @TODO this!
np.random.seed(0) # Random train/test splits stay the same between runs

# Divide the data into diff blocks
split_point = lambda x: int(len(x) * .80)

def train_split(x):
    return x[:split_point(x)]
def test_split(x):
    return x[split_point(x):]

train_pos_paths = train_split(pos_paths)
train_neg_paths = train_split(neg_paths)
train_questions = train_split(questions)

test_pos_paths = test_split(pos_paths)
test_neg_paths = test_split(neg_paths)
test_questions = test_split(questions)

neg_paths_per_epoch_train = 10
neg_paths_per_epoch_test = 10
dummy_y_train = np.zeros(len(train_questions)*neg_paths_per_epoch_train)
dummy_y_test = np.zeros(len(test_questions)*(neg_paths_per_epoch_test+1))

print train_questions.shape
print train_pos_paths.shape
print train_neg_paths.shape

print test_questions.shape
print test_pos_paths.shape
print test_neg_paths.shape

with K.tf.device('/gpu:' + gpu):
    from keras.models import load_model
    metric = rank_precision_metric(10)
    model = load_model("data/training/pairwise/model_56/model.h5", {'custom_loss':custom_loss, 'metric':metric})
    print rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000)

















