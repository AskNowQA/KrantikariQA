"""
    A file with MANY helper functions including losses etc which help with managing and implementing NNs.

    Every other code must have its own load_data, and model, and train code, and use as much of this as possible.

"""

# Shared Feature Extraction Layer
from __future__ import absolute_import
import os
import pickle
import sys
import json
import math
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
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers import Bidirectional, GRU, LSTM
from keras.models import Sequential, Model, model_from_json
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D

# Custom imports
from utils import embeddings_interface


# Some Macros
DEBUG = True

# Data locations
DATA_DIR = './data/models/core_chain/lcquad/'
CACHE_DATA_DIR = './data/data/core_chain/lcquad/'
RAW_DATA_DIR = './resources_v8/'

# Network Macros
MAX_SEQ_LENGTH = 25
EPOCHS = 300
BATCH_SIZE = 880        # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
NEGATIVE_SAMPLES = 1000
OPTIMIZER = optimizers.Adam(LEARNING_RATE)

np.random.seed(42)

# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


"""
    F1 measure functions
"""
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
        warnings.warn("Could not get model json")
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
    if DEBUG: print("@smart save model called")
    json_desc, l_dir = get_smart_save_path(model)
    path = os.path.join(l_dir, 'model.h5')
    if DEBUG:
        print("network.py:smart_save_model: Saving model in %s" % path)
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

    precision = float(len(np.where(np.argmax(outputs, axis=1) == 0)[0]))/outputs.shape[0]
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
        #     batch_neg_paths = neg_paths[np.random.randint(0, neg_paths.shape[0], BATCH_SIZE)]bay

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
                mode=_outer, output_shape=(self.max_length, self.max_length))


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
        self.model.add(Dense(nr_hidden, name='compare1', init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nr_hidden, name='compare2', W_regularizer=l2(L2), init='he_normal'))
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
                                          dropout_W=dropout,
                                          dropout_U=dropout),
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
        # tuning = merge([tuning, mod_sent],
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


def cross_correlation(x):
    a, b = x
    tf = K.tf
    a_fft = tf.fft(tf.complex(a, 0.0))
    b_fft = tf.fft(tf.complex(b, 0.0))
    ifft = tf.ifft(tf.conj(a_fft) * b_fft)
    return tf.cast(tf.real(ifft), 'float32')


def remove_positive_path(positive_path, negative_paths):
    counter = 0
    new_negative_paths = []
    for i in range(0, len(negative_paths)):
        if not np.array_equal(negative_paths[i], positive_path):
            new_negative_paths.append(np.asarray(negative_paths[i]))
        else:
            counter += 1
            print counter
    return new_negative_paths


def create_dataset(file, max_sequence_length, relations):
    """
        This file is meant to create data for core-chain ranking ONLY.

    :param file: id_big_data file
    :param max_sequence_length: for padding/cropping
    :param relations: the relations file to backtrack and look up shit.
    :return:
    """
    glove_embeddings = get_glove_embeddings()

    try:
        with open(os.path.join(CACHE_DATA_DIR, file + ".mapped.npz")) as data, open(os.path.join(CACHE_DATA_DIR, file + ".vocab.pickle")) as idx:
            dataset = np.load(data)
            questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
            vocab = pickle.load(idx)
            vectors = glove_embeddings[vocab.keys()]
            return vectors, questions, pos_paths, neg_paths
    except (EOFError,IOError) as e:
        with open(os.path.join(RAW_DATA_DIR, file)) as fp:
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
            for i in range(0,len(dataset)):
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
                negative_paths = remove_positive_path(pos_paths[i],negative_paths)
                try:
                    negative_paths = np.random.choice(negative_paths,1000)
                except ValueError:
                    if len(negative_paths) == 0:
                        negative_paths = neg_paths[-1]
                        print("Using previous question's paths for this since no neg paths for this question.")
                    else:
                        index = np.random.randint(0, len(negative_paths), 1000)
                        negative_paths = np.array(negative_paths)
                        negative_paths = negative_paths[index]
                neg_paths.append(negative_paths)

            for i in range(0,len(neg_paths)):
                neg_paths[i] = pad_sequences(neg_paths[i], maxlen=max_sequence_length, padding='post')
            neg_paths = np.asarray(neg_paths)
            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
            all = np.concatenate([questions, pos_paths, neg_paths.reshape((neg_paths.shape[0]*neg_paths.shape[1], neg_paths.shape[2]))], axis=0)

            # ###################
            # Map to new ID space
            # ###################

            uniques = np.unique(all)
            try:
                vocab = pickle.load(open(os.path.join(RAW_DATA_DIR, file + ".vocab.pickle")))
            except (IOError, EOFError) as e:
                if DEBUG:
                    warnings.warn("Did not find the vocabulary.")
                vocab = {}
                # Create Vocabulary
                for i in range(len(uniques)):
                    vocab[uniques[i]] = i

            # vocab = {}

            # Map everything
            for i in range(len(questions)):
                questions[i] = np.asarray([vocab[key] for key in questions[i]])
                pos_paths[i] = np.asarray([vocab[key] for key in pos_paths[i]])

                for j in range(len(neg_paths[i])):
                    neg_paths[i][j] = np.asarray([vocab[key] for key in neg_paths[i][j]])

            # Create slimmer, better, faster, vectors file.
            vectors = glove_embeddings[uniques]

            with open(os.path.join(CACHE_DATA_DIR, file + ".mapped.npz"), "w") as data, open(os.path.join(RAW_DATA_DIR, file + ".vocab.pickle"), "w") as idx:
                np.savez(data, questions, pos_paths, neg_paths)
                pickle.dump(vocab,idx)

            return vectors, questions, pos_paths, neg_paths


if __name__ == "__main__":
    warnings.warn("Code has been moved from this file to network_corechain. Please open that instead. Okay, auf weidersehen")