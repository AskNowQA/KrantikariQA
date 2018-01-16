from __future__ import print_function

import os
import numpy as np

from abc import abstractmethod
from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Conv1D, Lambda, LSTM, Dense, concatenate, TimeDistributed, \
    GlobalAveragePooling1D, GlobalMaxPooling1D
from keras import backend as K
from keras.models import Model
import keras


class QARankingModel:
    '''
    Abstract class for pairwise ranking based question answering models
    '''
    def __init__(self, max_sequence_length, data_dir, similarity='dot', dropout=0.5):
        self._data_dir = data_dir

        self.question = Input(shape=(max_sequence_length,), dtype='int32', name='question_input')
        self.pos_path = Input(shape=(max_sequence_length,), dtype='int32', name='pos_path_input')
        self.neg_path = Input(shape=(max_sequence_length,), dtype='int32', name='neg_path_input')
        self._path = Input(shape=(max_sequence_length,), dtype='int32', name='path_stub')

        self.dropout = dropout
        self.similarity = similarity
        self.max_sequence_length = max_sequence_length

        # initialize a bunch of variables that will be set later
        self._question_score = None
        self._path_score = None
        self._similarities = None
        self._score_model = None

        self.training_model = None
        self.prediction_model = None

    @abstractmethod
    def build(self):
        return

    def get_similarity(self):
        '''
        Specify similarity in configuration under 'similarity' -> 'mode'
        If a parameter is needed for the model, specify it in 'similarity'
        Example configuration:
        config = {
            ... other parameters ...
            'similarity': {
                'mode': 'gesd',
                'gamma': 1,
                'c': 1,
            }
        }
        cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
        polynomial: (gamma * dot(a, b) + c) ^ d
        sigmoid: tanh(gamma * dot(a, b) + c)
        rbf: exp(-gamma * l2_norm(a-b) ^ 2)
        euclidean: 1 / (1 + l2_norm(a - b))
        exponential: exp(-gamma * l2_norm(a - b))
        gesd: euclidean * sigmoid
        aesd: (euclidean + sigmoid) / 2
        '''

        params = self.params
        similarity = params['mode']

        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        l2_norm = lambda a, b: K.sqrt(K.sum(K.square(a - b), axis=1, keepdims=True))

        if similarity == 'dot':
            return lambda x: dot(x[0], x[1])
        elif similarity == 'cosine':
            return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())
        elif similarity == 'polynomial':
            return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
        elif similarity == 'sigmoid':
            return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
        elif similarity == 'rbf':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
        elif similarity == 'euclidean':
            return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        elif similarity == 'exponential':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
        elif similarity == 'gesd':
            euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) * sigmoid(x)
        elif similarity == 'aesd':
            euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) + sigmoid(x)
        else:
            raise Exception('Invalid similarity: {}'.format(similarity))

    def get_score_model(self):
        if None in [self._question_score, self._path_score]:
            self._question_score, self._path_score = self.build()

        if self._score_model is None:
            dropout = Dropout(self.dropout)
            similarity = self.get_similarity()

            qa_model = Lambda(similarity, output_shape=lambda _: (None, 1))([dropout(self._question_score),
                                                                             dropout(self._path_score)])
            self._score_model = Model(inputs=[self.question, self._path], outputs=qa_model, name='qa_model')

        return self._score_model

    def get_training_model(self):
        if not self.training_model:
            score_model = self.get_score_model()
            pos_score = score_model([self.question, self.pos_path])
            neg_score = score_model([self.question, self.neg_path])
            loss = Lambda(lambda x: K.relu(self.config['margin'] - x[0] + x[1]),
                      output_shape=lambda x: x[0])([pos_score, neg_score])
            self.training_model = Model(inputs=[self.question, self.pos_path, self.neg_path], outputs=loss,
                                    name='training_model')
        return self.training_model

    def get_prediction_model(self):
        if not self.prediction_model:
            score_model = self.get_score_model()
            score = score_model([self.question, self.pos_path])
            self.prediction_model = Model(inputs=[self.question, self.pos_path], outputs=score,
                                          name='prediction_model')
        return self.prediction_model


    def compile(self, optimizer, **kwargs):
        self.prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

    def fit(self, training_data, validation_data, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        assert isinstance(self.prediction_model, Model)
        return self.training_model.fit_generator(training_data, validation_data=validation_data, **kwargs)

    def predict(self, x, **kwargs):
        assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
        return self.prediction_model.predict(x, **kwargs)

    def get_next_save_path(self, **kwargs):
        # Find the current model dirs in the data dir.
        _, dirs, _ = os.walk(self._data_dir).next()

        # If no folder found in there, create a new one.
        if len(dirs) == 0:
            os.mkdir(os.path.join(self._data_dir, "model_00"))
            dirs = ["model_00"]

        # Find the latest folder in here
        dir_nums = sorted([ x[-2:] for x in dirs])
        l_dir = os.path.join(self._data_dir, "model_" + dir_nums[-1])

        # Create new folder with name model_(i+1)

        new_num = int(dir_nums[-1]) + 1
        if new_num < 10:
            new_num = str('0') + str(new_num)
        else:
            new_num = str(new_num)

        l_dir = os.path.join(self._data_dir, "model_" + new_num)
        os.mkdir(l_dir)

        return l_dir

    def save_model(self, path):
        self.prediction_model.save(path)

    def load_model(self, path):
        self.prediction_model = keras.models.load_model(path)
