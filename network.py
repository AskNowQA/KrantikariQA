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
from pprint import pprint

from utils import embeddings_interface



# Some Macros
DEBUG = True
DATA_DIR = './data/training/pairwise'
RESOURCE_DIR = './resources_v8'
EPOCHS = 300
BATCH_SIZE = 880 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
NEGATIVE_SAMPLES = 1000
OPTIMIZER = optimizers.Adam(LEARNING_RATE)

'''
    F1 measure functions
'''
def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return true_positives


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))



def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def get_smart_save_path(model):
    desc = None
    try:
        # Get the model description
        desc = model.to_json()
    except TypeError:
        print "Could not get model json"
        pass

    # Find the current model dirs in the data dir.
    _, dirs, _ = os.walk(DATA_DIR).next()

    # If no folder found in there, create a new one.
    if len(dirs) == 0:
        os.mkdir(os.path.join(DATA_DIR, "model_00"))
        dirs = ["model_00"]

    # Find the latest folder in here
    dir_nums = sorted([ x[-2:] for x in dirs])
    l_dir = os.path.join(DATA_DIR, "model_" + dir_nums[-1])

    # Check if the latest dir has the same model as current
    try:
        if json.load(open(os.path.join(l_dir, 'model.json'))) != desc:
            # Diff model. Make new folder and do stuff. @TODO this
            new_num = int(dir_nums[-1]) + 1
            if new_num < 10:
                new_num = str('0') + str(new_num)
            else:
                new_num = str(new_num)

            l_dir = os.path.join(DATA_DIR, "model_" + new_num)
            os.mkdir(l_dir)
    except:
        pass
    finally:
        return desc, l_dir


def smart_save_model(model):
    """
        Function to properly save the model to disk.
            If the model config is the same as one already on disk, overwrite it.
            Else make a new folder and write things there

    :return: None
    """
    if DEBUG: print "@smart save model called"
    json_desc, l_dir = get_smart_save_path(model)
    path = os.path.join(l_dir, 'model.h5')
    if DEBUG:
        print "network.py:smart_save_model: Saving model in %s" % path
    model.save(path)
    json.dump(json_desc, open(os.path.join(l_dir, 'model.json'), 'w+'))

def zeroloss(yt, yp):
    return 0.0

def custom_loss(y_true, y_pred):
    '''
        max margin loss
    '''
    # y_pos = y_pred[0]
    # y_neg= y_pred[1]
    diff = y_pred[:,-1]
    # return K.sum(K.maximum(1.0 - diff, 0.))
    return K.sum(diff)

def load_relation(relation_file):
    relations = pickle.load(open(relation_file))
    inverse_relations = {}
    for key in relations:
        value = relations[key]
        new_key = value[0]
        value[0] = key
        inverse_relations[new_key] = value

    return inverse_relations

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

class CustomModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, test_questions, test_pos_paths, test_neg_paths, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.test_questions = test_questions
        self.test_pos_paths = test_pos_paths
        self.test_neg_paths = test_neg_paths

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = rank_precision(self.model, self.test_questions, self.test_pos_paths, self.test_neg_paths, 1000, 10000)
                print('\Validation recall@1: {}\n'.format(current))
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            smart_save_model(self.model)
                            # self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    smart_save_model(self.model)
                    # self.model.save(filepath, overwrite=True)


class TrainingDataGenerator(Sequence):
    def __init__(self, questions, pos_paths, neg_paths, max_length, neg_paths_per_epoch, batch_size):
        self.dummy_y = np.zeros(batch_size)
        self.firstDone = False
        self.max_length = max_length
        self.neg_paths_per_epoch = neg_paths_per_epoch

        self.questions = np.reshape(np.repeat(np.reshape(questions,
                                            (questions.shape[0], 1, questions.shape[1])),
                                 neg_paths_per_epoch, axis=1), (-1, max_length))

        self.pos_paths = np.reshape(np.repeat(np.reshape(pos_paths,
                                            (pos_paths.shape[0], 1, pos_paths.shape[1])),
                                 neg_paths_per_epoch, axis=1), (-1, max_length))

        self.neg_paths = neg_paths

        self.neg_paths_sampled = np.reshape(self.neg_paths[:,np.random.randint(0, NEGATIVE_SAMPLES, self.neg_paths_per_epoch), :],
                                            (-1, self.max_length))

        self.questions_shuffled, self.pos_paths_shuffled, self.neg_paths_shuffled = \
            shuffle(self.questions, self.pos_paths, self.neg_paths_sampled)



        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.questions) / self.batch_size)

    def __getitem__(self, idx):
        index = lambda x: x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_questions = index(self.questions_shuffled)
        batch_pos_paths = index(self.pos_paths_shuffled)
        batch_neg_paths = index(self.neg_paths_shuffled)

        # if self.firstDone == False:
        #     batch_neg_paths = index(self.neg_paths)
        # else:
        #     batch_neg_paths = neg_paths[np.random.randint(0, neg_paths.shape[0], BATCH_SIZE)]


        return ([batch_questions, batch_pos_paths, batch_neg_paths], self.dummy_y)

    def on_epoch_end(self):
        self.firstDone = not self.firstDone
        self.neg_paths_sampled = np.reshape(self.neg_paths[:,np.random.randint(0, NEGATIVE_SAMPLES, self.neg_paths_per_epoch), :],
                                            (-1, self.max_length))
        self.questions_shuffled, self.pos_paths_shuffled, self.neg_paths_shuffled = \
            shuffle(self.questions, self.pos_paths, self.neg_paths_sampled)


class ValidationDataGenerator(Sequence):
    def __init__(self, questions, pos_paths, neg_paths, max_length, neg_paths_per_epoch, batch_size):
        self.dummy_y = np.zeros(batch_size)
        self.firstDone = False
        self.max_length = max_length
        self.neg_paths_per_epoch = neg_paths_per_epoch

        self.questions = np.reshape(np.repeat(np.reshape(questions,
                                            (questions.shape[0], 1, questions.shape[1])),
                                 neg_paths_per_epoch+1, axis=1), (-1, max_length))

        self.pos_paths = np.reshape(pos_paths,
                                            (pos_paths.shape[0], 1, pos_paths.shape[1]))
        self.neg_paths = neg_paths
        neg_paths_sampled = self.neg_paths[:, np.random.randint(0, NEGATIVE_SAMPLES, self.neg_paths_per_epoch), :]
        self.all_paths = np.reshape(np.concatenate([self.pos_paths, neg_paths_sampled], axis=1), (-1, self.max_length))

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.questions) / self.batch_size)

    def __getitem__(self, idx):
        index = lambda x: x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_questions = index(self.questions)
        batch_all_paths = index(self.all_paths)

        # if self.firstDone == False:
        #     batch_neg_paths = index(self.neg_paths)
        # else:
        #     batch_neg_paths = neg_paths[np.random.randint(0, neg_paths.shape[0], BATCH_SIZE)]


        return ([batch_questions, batch_all_paths, np.zeros_like(batch_all_paths)], self.dummy_y)

    def on_epoch_end(self):
        self.firstDone = not self.firstDone
        neg_paths_sampled = self.neg_paths[:, np.random.randint(0, NEGATIVE_SAMPLES, self.neg_paths_per_epoch), :]
        self.all_paths = np.reshape(np.concatenate([self.pos_paths, neg_paths_sampled], axis=1), (-1, self.max_length))


def rank_precision_metric(neg_paths_per_epoch):
    def metric(y_true, y_pred):
        scores = y_pred[:, 0]
        scores = K.reshape(scores, (-1, neg_paths_per_epoch+1))
        hits = K.cast(K.shape(K.tf.where(K.tf.equal(K.tf.argmax(scores, axis=1),0)))[0], 'float32')
        precision = hits/K.cast(K.shape(scores)[0], 'float32')
        # precision = float(len(np.where(np.argmax(all_outputs, axis=1)==0)[0]))/all_outputs.shape[0]
        return precision
    return metric

class _Attention(object):
    def __init__(self, max_length, nr_hidden, dropout=0.0, L2=0.0, activation='relu'):
        self.max_length = max_length
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(nr_hidden,)))
        self.model.add(
            Dense(nr_hidden, name='attend1',
                init='he_normal', W_regularizer=l2(L2),
                input_shape=(nr_hidden,), activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nr_hidden, name='attend2',
            init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model = TimeDistributed(self.model)

    def __call__(self, sent1, sent2):
        def _outer(AB):
            att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
            return K.permute_dimensions(att_ji,(0, 2, 1))
        return merge(
                [self.model(sent1), self.model(sent2)],
                mode=_outer,
                output_shape=(self.max_length, self.max_length))


class _SoftAlignment(object):
    def __init__(self, max_length, nr_hidden):
        self.max_length = max_length
        self.nr_hidden = nr_hidden

    def __call__(self, sentence, attention, transpose=False):
        def _normalize_attention(attmat):
            att = attmat[0]
            mat = attmat[1]
            if transpose:
                att = K.permute_dimensions(att,(0, 2, 1))
            # 3d softmax
            e = K.exp(att - K.max(att, axis=-1, keepdims=True))
            s = K.sum(e, axis=-1, keepdims=True)
            sm_att = e / s
            return K.batch_dot(sm_att, mat)
        return merge([attention, sentence], mode=_normalize_attention,
                      output_shape=(self.max_length, self.nr_hidden)) # Shape: (i, n)


class _Comparison(object):
    def __init__(self, words, nr_hidden, L2=0.0, dropout=0.0):
        self.words = words
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(nr_hidden*2,)))
        self.model.add(Dense(nr_hidden, name='compare1',
            init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nr_hidden, name='compare2',
                        W_regularizer=l2(L2), init='he_normal'))
        self.model.add(Activation('relu'))
        self.model = TimeDistributed(self.model)

    def __call__(self, sent, align, **kwargs):
        result = self.model(merge([sent, align], mode='concat')) # Shape: (i, n)
        avged = GlobalAveragePooling1D()(result)
        maxed = GlobalMaxPooling1D()(result)
        merged = merge([avged, maxed])
        result = BatchNormalization()(merged)
        return result


class _Entailment(object):
    def __init__(self, nr_hidden, nr_out, dropout=0.0, L2=0.0):
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(nr_hidden*2,)))
        self.model.add(Dense(nr_hidden, name='entail1',
            init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(dropout))
        self.model.add(Dense(nr_out, name='entail_out',
            init='he_normal', W_regularizer=l2(L2)))
        # self.model.add(Activation('relu'))
        # self.model.add(Dense(nr_out, name='entail_out', activation='softmax',
        #                 W_regularizer=l2(L2), init='zero'))

    def __call__(self, feats1, feats2):
        features = merge([feats1, feats2], mode='concat')
        return self.model(features)

class _GlobalSumPooling1D(Layer):
    '''Global sum pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''
    def __init__(self, **kwargs):
        super(_GlobalSumPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if mask is not None:
            return K.sum(x * K.clip(mask, 0, 1), axis=1)
        else:
            return K.sum(x, axis=1)

class _BiRNNEncoding(object):
    def __init__(self, max_length, embedding_dims, units, dropout=0.0):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units, return_sequences=True,
                                         dropout_W=dropout, dropout_U=dropout),
                                         input_shape=(max_length, embedding_dims)))
        #self.model.add(LSTM(units, return_sequences=False,
        #                                 dropout_W=dropout, dropout_U=dropout))
        self.model.add(TimeDistributed(Dense(units, activation='relu', init='he_normal')))
        self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, sentence):
        return self.model(sentence)

class _simple_BiRNNEncoding(object):
    def __init__(self, max_length, embedding_dims, units, dropout=0.0):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units, return_sequences=False,
                                         dropout_W=dropout, dropout_U=dropout),
                                         input_shape=(max_length, embedding_dims)))
        #self.model.add(LSTM(units, return_sequences=False,
        #                                 dropout_W=dropout, dropout_U=dropout))
        # self.model.add(TimeDistributed(Dense(units, activation='relu', init='he_normal')))
        # self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, sentence):
        return self.model(sentence)

class _StaticEmbedding(object):
    def __init__(self, vectors, max_length, nr_out, nr_tune=5000, dropout=0.0):
        self.nr_out = nr_out
        self.max_length = max_length
        self.embed = Embedding(
                        vectors.shape[0],
                        vectors.shape[1],
                        input_length=max_length,
                        weights=[vectors],
                        name='embed',
                        trainable=True,)
        self.tune = Embedding(
                        nr_tune,
                        nr_out,
                        input_length=max_length,
                        weights=None,
                        name='tune',
                        trainable=True,
                        dropout=dropout)
        self.mod_ids = Lambda(lambda sent: sent % (nr_tune-1)+1,
                              output_shape=(self.max_length,))

        self.project = TimeDistributed(
                            Dense(
                                nr_out,
                                activation=None,
                                bias=False,
                                name='project'))

    def __call__(self, sentence):
        mod_sent = self.mod_ids(sentence)
        tuning = self.tune(mod_sent)
        #tuning = merge([tuning, mod_sent],
        #    mode=lambda AB: AB[0] * (K.clip(K.cast(AB[1], 'float32'), 0, 1)),
        #    output_shape=(self.max_length, self.nr_out))
        pretrained = self.project(self.embed(sentence))
        vectors = merge([pretrained, tuning], mode='sum')
        return vectors

def get_glove_embeddings():
    from utils.embeddings_interface import __check_prepared__
    __check_prepared__('glove')

    from utils.embeddings_interface import glove_embeddings
    return glove_embeddings

class ValidationCallback(Callback):
    def __init__(self, test_data, test_questions, test_pos_paths, test_neg_paths):
        self.test_data = test_data
        self.test_questions = test_questions
        self.test_pos_paths = test_pos_paths
        self.test_neg_paths = test_neg_paths

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 20 == 0:
            recall = rank_precision(self.model, self.test_questions, self.test_pos_paths, self.test_neg_paths, 1000, 10000)
            print('\Validation recall@1: {}\n'.format(recall))

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

def load_data(file, max_sequence_length, relations):
    glove_embeddings = get_glove_embeddings()

    try:
        with open(os.path.join(RESOURCE_DIR, file + ".mapped.npz")) as data, open(os.path.join(RESOURCE_DIR, file + ".index.npy")) as idx:
            dataset = np.load(data)
            # dataset = dataset[:10]
            questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
            index = np.load(idx)
            vectors = glove_embeddings[index]
            return vectors, questions, pos_paths, neg_paths
    except (EOFError,IOError) as e:
        with open(os.path.join(RESOURCE_DIR, file)) as fp:
            dataset = json.load(fp)
            dataset = dataset[:10]
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
            for i in dataset:
                negative_paths_id = i['uri']['hop-2-properties'] + i['uri']['hop-1-properties']
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
                try:
                    negative_paths = np.random.choice(negative_paths,1000)
                except ValueError:
                    if len(negative_paths) == 0:
                        negative_paths = neg_paths[-1]
                        print "Using previous question's paths for this since no neg paths for this question."
                    index = np.random.randint(0, len(negative_paths), 1000)
                    negative_paths = np.array(negative_paths)
                    negative_paths = negative_paths[index]
                neg_paths.append(negative_paths)

            # neg_paths = [i[2] for i in dataset]
            #####################
            #Removing duplicates#
            #####################
            temp = neg_paths[0][0]
            for i in xrange(0, len(pos_paths)):
                to_remove = []
                for j in range(0,len(neg_paths[i])):
                    if np.array_equal(pos_paths[i], neg_paths[i][j]):
                        # to_remove.append(j)
                        if j != 0:
                            if not np.array_equal(pos_paths[i], neg_paths[i][j-1]):
                                neg_paths[i][j] = neg_paths[i][j-1]
                            else:
                                if j- 2 != 0:
                                    neg_paths[i][j] = neg_paths[i][j-2]
                                else:
                                    try:
                                        neg_paths[i][j] = neg_paths[i][j+1]
                                    except IndexError:
                                        neg_paths[i][j] = neg_paths[i][j-1]
                        else:
                            if not np.array_equal(pos_paths[i], neg_paths[i][j + 1]):
                                neg_paths[i][j] = neg_paths[i][j+1]
                            else:
                                try:
                                    neg_paths[i][j] = neg_paths[i][j+2]
                                except IndexError:
                                    neg_paths[i][j] = neg_paths[i][j+1]


                # neg_paths[i] = np.delete(neg_paths[i], to_remove) if to_remove else neg_paths[i]
            # for index in to_remove:


            neg_paths = [path for paths in neg_paths for path in paths]
            neg_paths = pad_sequences(neg_paths, maxlen=max_sequence_length, padding='post')
            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
            pickle.dump(pos_paths,open('./resources_v8/pos_before.pickle','w+'))
            all = np.concatenate([questions, pos_paths, neg_paths], axis=0)
            mapped_all, index = pd.factorize(all.flatten(), sort=True)
            pickle.dump(index,open('./resources_v8/index.pickle','w+'))
            pickle.dump(mapped_all,open('./resources_v8/mapped_all.pickle','w+'))
            mapped_all = mapped_all.reshape((-1, max_sequence_length))
            vectors = glove_embeddings[index]

            questions, pos_paths, neg_paths = np.split(mapped_all, [questions.shape[0], questions.shape[0]*2])
            pickle.dump(pos_paths, open('./resources_v8/pos_after.pickle','w+'))
            neg_paths = np.reshape(neg_paths, (len(questions), NEGATIVE_SAMPLES, max_sequence_length))

            with open(os.path.join(RESOURCE_DIR, file + ".mapped.npz"), "w") as data, open(os.path.join(RESOURCE_DIR, file + ".index.npy"), "w") as idx:
                np.savez(data, questions, pos_paths, neg_paths)
                np.save(idx, index)

            return vectors, questions, pos_paths, neg_paths

def main_parikh(_gpu):

    gpu = _gpu
    """
        Data Time!
    """
    # Pull the data up from disk
    max_length = 25
    relations = load_relation('resources_v8/relations.pickle')
    vectors, questions, pos_paths, neg_paths = load_data("id_big_data.json", max_length,relations)

    counter = 0
    for i in xrange(0, len(pos_paths)):
        temp = -1
        for j in xrange(0,len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j+10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    # if counter > 0:
    #     print counter
    #     print "critical condition entered"
    #     pickle.dump(questions,open('./resources_v8/questions_new.pickle','w+'))
    #     pickle.dump(vectors,open('./resources_v8/vectors_new.pickle','w+'))
    #     pickle.dump(pos_paths,open('./resources_v8/positive_paths_new.pickle','w+'))
    #     pickle.dump(neg_paths,open('./resources_v8/negative_paths_new.pickle','w+'))
    #     print "done with dumping everything"
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
    neg_paths_per_epoch_test = 1000
    dummy_y_train = np.zeros(len(train_questions)*neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions)*(neg_paths_per_epoch_test+1))

    print train_questions.shape
    print train_pos_paths.shape
    print train_neg_paths.shape

    print test_questions.shape
    print test_pos_paths.shape
    print test_neg_paths.shape

    with K.tf.device('/gpu:' + gpu):
        neg_paths_per_epoch_train = 10
        neg_paths_per_epoch_test = 1000
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True)))
        """
            Model Time!
        """
        max_length = train_questions.shape[1]
        # Define input to the models
        x_ques = Input(shape=(max_length,), dtype='int32', name='x_ques')
        x_pos_path = Input(shape=(max_length,), dtype='int32', name='x_pos_path')
        x_neg_path = Input(shape=(max_length,), dtype='int32', name='x_neg_path')

        embedding_dims = vectors.shape[1]
        nr_hidden = 128

        # holographic_forward = Dense(1, activation='sigmoid')
        # final_forward = Dense(1, activation='sigmoid')

        embed = _StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = _BiRNNEncoding(max_length, embedding_dims,  nr_hidden, 0.5)
        # encode = LSTM(max_length)(encode)
        attend = _Attention(max_length, nr_hidden, dropout=0.6, L2=0.01)
        align = _SoftAlignment(max_length, nr_hidden)
        compare = _Comparison(max_length, nr_hidden, dropout=0.6, L2=0.01)
        entail = _Entailment(nr_hidden, 1, dropout=0.4, L2=0.01)

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # holographic_score = holographic_forward(Lambda(lambda x: cross_correlation(x)) ([ques_encoded, path_encoded]))
            # dot_score = dot([ques_encoded, path_encoded], axes=-1)
            # l1_score = Lambda(lambda x: K.abs(x[0]-x[1]))([ques_encoded, path_encoded])

            # return final_forward(concatenate([holographic_score, dot_score, l1_score], axis=-1))
            # return dot_score

            #
            attention = attend(ques_encoded, path_encoded)

            align_ques = align(path_encoded, attention)
            align_path = align(ques_encoded, attention, transpose=True)

            feats_ques = compare(ques_encoded, align_ques)
            feats_path = compare(path_encoded, align_path)

            return entail(feats_ques, feats_path)

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
            outputs=[output])

        print(model.summary())

        model.compile(optimizer=OPTIMIZER,
                      loss=custom_loss)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_train, BATCH_SIZE)
        validation_generator = ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,\
            monitor='val_metric', verbose=1, save_best_only=True, mode='max', period=10)

        model.fit_generator(training_generator, epochs=EPOCHS, workers=3, use_multiprocessing=True, callbacks=[checkpointer])
            # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    # ])



        # Prepare test data

        print "Precision (hits@1) = ", rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000)

    # print "Evaluation Complete"
    # print "Loss     = ", results[0]
    # print "F1 Score = ", results[1]
    # print "Accuracy = ", results[2]


def main_bidirectional(_gpu):


    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = 25
    relations = load_relation('resources_v8/relations.pickle')
    vectors, questions, pos_paths, neg_paths = load_data("id_big_data.json", max_length,relations)

    counter = 0
    for i in xrange(0, len(pos_paths)):
        temp = -1
        for j in xrange(0,len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j+10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    # if counter > 0:
    #     print counter
    #     print "critical condition entered"
    #     pickle.dump(questions,open('./resources_v8/questions_new.pickle','w+'))
    #     pickle.dump(vectors,open('./resources_v8/vectors_new.pickle','w+'))
    #     pickle.dump(pos_paths,open('./resources_v8/positive_paths_new.pickle','w+'))
    #     pickle.dump(neg_paths,open('./resources_v8/negative_paths_new.pickle','w+'))
    #     print "done with dumping everything"
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
    neg_paths_per_epoch_test = 1000
    dummy_y_train = np.zeros(len(train_questions)*neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions)*(neg_paths_per_epoch_test+1))

    print train_questions.shape
    print train_pos_paths.shape
    print train_neg_paths.shape

    print test_questions.shape
    print test_pos_paths.shape
    print test_neg_paths.shape

    with K.tf.device('/gpu:' + gpu):
        neg_paths_per_epoch_train = 10
        neg_paths_per_epoch_test = 1000
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True)))
        """
            Model Time!
        """
        max_length = train_questions.shape[1]
        # Define input to the models
        x_ques = Input(shape=(max_length,), dtype='int32', name='x_ques')
        x_pos_path = Input(shape=(max_length,), dtype='int32', name='x_pos_path')
        x_neg_path = Input(shape=(max_length,), dtype='int32', name='x_neg_path')

        embedding_dims = vectors.shape[1]
        nr_hidden = 128

        # holographic_forward = Dense(1, activation='sigmoid')
        # final_forward = Dense(1, activation='sigmoid')

        embed = _StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = _simple_BiRNNEncoding(max_length, embedding_dims,  nr_hidden, 0.5)
        # encode = LSTM(max_length)(encode)
        # attend = _Attention(max_length, nr_hidden, dropout=0.6, L2=0.01)
        # align = _SoftAlignment(max_length, nr_hidden)
        # compare = _Comparison(max_length, nr_hidden, dropout=0.6, L2=0.01)
        # entail = _Entailment(nr_hidden, 1, dropout=0.4, L2=0.01)

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # holographic_score = holographic_forward(Lambda(lambda x: cross_correlation(x)) ([ques_encoded, path_encoded]))
            dot_score = dot([ques_encoded, path_encoded], axes=-1)
            # l1_score = Lambda(lambda x: K.abs(x[0]-x[1]))([ques_encoded, path_encoded])

            # return final_forward(concatenate([holographic_score, dot_score, l1_score], axis=-1))
            return dot_score

            #
            # attention = attend(ques_encoded, path_encoded)
			#
            # align_ques = align(path_encoded, attention)
            # align_path = align(ques_encoded, attention, transpose=True)
			#
            # feats_ques = compare(ques_encoded, align_ques)
            # feats_path = compare(path_encoded, align_path)

            # return entail(feats_ques, feats_path)

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
            outputs=[output])

        print(model.summary())

        model.compile(optimizer=OPTIMIZER,
                      loss=custom_loss)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_train, BATCH_SIZE)
        validation_generator = ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,\
            monitor='val_metric', verbose=1, save_best_only=True, mode='max', period=10)

        model.fit_generator(training_generator, epochs=EPOCHS, workers=3, use_multiprocessing=True, callbacks=[checkpointer])
            # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    # ])



        # Prepare test data

        print "Precision (hits@1) = ", rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000)

if __name__ == "__main__":
    gpu = sys.argv[1]
    model = sys.argv[2]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if model == "p":
        main_parikh(gpu)
    else:
        main_bidirectional(gpu)
