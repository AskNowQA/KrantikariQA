"""
    The file with different models for corechain ranking.
    bidirectional_dot

"""

from __future__ import absolute_import
import os
import sys
import json
import warnings
import numpy as np

import keras.backend.tensorflow_backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Add

import prepare_transfer_learning
import network as n

# Macros
DEBUG = True
CHECK_VALIDATION_ACC_PERIOD = 10
LCQUAD_BIRNN_MODEL = 'model_14'


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def bidirectional_dot_sigmoidloss(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                                  _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None):
    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index:
        split_point = index + 1
    else:
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

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = n._simple_BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5)

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # holographic_score = holographic_forward(Lambda(lambda x: cross_correlation(x)) ([ques_encoded, path_encoded]))
            dot_score = n.dot([ques_encoded, path_encoded], axes=-1)
            # l1_score = Lambda(lambda x: K.abs(x[0]-x[1]))([ques_encoded, path_encoded])

            # return final_forward(concatenate([holographic_score, dot_score, l1_score], axis=-1))
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: 1.0 - K.sigmoid(x[0] - x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                     max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                         max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=CHECK_VALIDATION_ACC_PERIOD)

        model.fit_generator(training_generator,
                            epochs=n.EPOCHS,
                            workers=3,
                            use_multiprocessing=True,
                            callbacks=[checkpointer])
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') ])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))


def bidirectional_dot(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                      _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None,rdf=False) :
    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index: split_point = lambda x: _index+1
    elif rdf: split_point = lambda x: int(len(x) * .90)
    else: split_point = lambda x: int(len(x) * .70)

    # print _index
    # print "shape of pos path is ", str(pos_paths.shape)
    def train_split(x):
        return x[:split_point(x)]

    def test_split(x):
        if _index or rdf:
            return x[split_point(x):]
        else:
            return x[split_point(x):int(.80 * len(x))]
        # return x[split_point(x):int(.80*len(x))]

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

    with K.tf.device('/gpu:' + gpu):
        neg_paths_per_epoch_train = 10
        neg_paths_per_epoch_test = 1000
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True)))
        """
            Model Time!
        """
        # max_length = train_questions.shape[1]
        # Define input to the models
        x_ques = Input(shape=(max_length,), dtype='int32', name='x_ques')
        x_pos_path = Input(shape=(max_length,), dtype='int32', name='x_pos_path')
        x_neg_path = Input(shape=(max_length,), dtype='int32', name='x_neg_path')

        embedding_dims = vectors.shape[1]
        nr_hidden = 256

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.3)
        encode = n._simple_BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5, _name="encoder")

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # holographic_score = holographic_forward(Lambda(lambda x: cross_correlation(x)) ([ques_encoded, path_encoded]))
            dot_score = n.dot([ques_encoded, path_encoded], axes=-1)
            # l1_score = Lambda(lambda x: K.abs(x[0]-x[1]))([ques_encoded, path_encoded])

            # return final_forward(concatenate([holographic_score, dot_score, l1_score], axis=-1))
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                     max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                         max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=CHECK_VALIDATION_ACC_PERIOD)

        model.fit_generator(training_generator,
                            epochs=n.EPOCHS,
                            workers=3,
                            use_multiprocessing=True,
                            callbacks=[checkpointer])
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') ])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))

def bidirectional_dot_pointwise(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                      _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None) :
    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index: split_point = lambda x: _index+1
    else: split_point = lambda x: int(len(x) * .70)

    # print _index
    # print "shape of pos path is ", str(pos_paths.shape)
    def train_split(x):
        return x[:split_point(x)]

    def test_split(x):
        if _index:
            return x[split_point(x):]
        else:
            return x[split_point(x):int(.80 * len(x))]
        # return x[split_point(x):int(.80*len(x))]

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

    with K.tf.device('/gpu:' + gpu):
        neg_paths_per_epoch_train = 10
        neg_paths_per_epoch_test = 1000
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True)))
        """
            Model Time!
        """
        # max_length = train_questions.shape[1]
        # Define input to the models
        x_ques = Input(shape=(max_length,), dtype='int32', name='x_ques')
        x_pos_path = Input(shape=(max_length,), dtype='int32', name='x_pos_path')
        x_neg_path = Input(shape=(max_length,), dtype='int32', name='x_neg_path')

        embedding_dims = vectors.shape[1]
        nr_hidden = 256

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.3)
        encode = n._simple_BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5, _name="encoder")

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # holographic_score = holographic_forward(Lambda(lambda x: cross_correlation(x)) ([ques_encoded, path_encoded]))
            dot_score = n.dot([ques_encoded, path_encoded], axes=-1)
            # l1_score = Lambda(lambda x: K.abs(x[0]-x[1]))([ques_encoded, path_encoded])

            # return final_forward(concatenate([holographic_score, dot_score, l1_score], axis=-1))
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        # loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        loss_1 = Lambda(lambda x: K.mean(K.categorical_crossentropy(np.ones_like(x[0]), x[0]), axis=-1))([pos_score])
        loss_2 = Lambda(lambda x: K.mean(K.categorical_crossentropy(np.zeros_like(x[0]), x[0]), axis=-1))([neg_score])

        loss = Add()([loss_1,loss_2])
        # loss = Lambda(lambda x: K.(x[0] , x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                     max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                         max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=CHECK_VALIDATION_ACC_PERIOD)

        model.fit_generator(training_generator,
                            epochs=n.EPOCHS,
                            workers=3,
                            use_multiprocessing=True,
                            callbacks=[checkpointer])
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') ])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))

def two_bidirectional_dot(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                      _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None):
    """
        A bi-lstm encodes the input.
        Another bi-lstm encodes the op
        siamese setup
        dot
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("Critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index: split_point = lambda x: _index+1
    else: split_point = lambda x: int(len(x) * .70)

    def train_split(x):
        return x[:split_point(x)]

    def test_split(x):
        if _index:
            return x[split_point(x):]
        return x[split_point(x):int(.80 * len(x))]

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode_one = n._double_BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.4, True, _name="double_encoder")
        # encode_two = n._BiRNNEncoding(max_length, nr_hidden*2, nr_hidden/2, 0.4)

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode_one(x_ques_embedded)
            path_encoded = encode_one(x_path_embedded)

            dot_score = n.dot([ques_encoded, path_encoded], axes=-1)

            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                     max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                         max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=CHECK_VALIDATION_ACC_PERIOD)

        model.fit_generator(training_generator,
                            epochs=n.EPOCHS,
                            workers=3,
                            use_multiprocessing=True,
                            callbacks=[checkpointer])
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') ])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))


def bidirectional_dense_dot(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                            _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None):
    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index:
        split_point = index + 1
    else:
        split_point = lambda x: int(len(x) * .70)

    def train_split(x):
        return x[:split_point(x)]

    def test_split(x):
        if _index:
            return x[split_point(x):]
        return x[split_point(x):int(.80 * len(x))]

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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
        nr_hidden = 256

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        # encode = n._BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5)
        encode = n._simple_BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5, return_sequences=False)
        dense = n._simpleDense(max_length, nr_hidden)

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            ques_dense = dense(ques_encoded)
            path_dense = dense(path_encoded)

            dot_score = n.dot([ques_dense, path_dense],axes = -1)
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())
        # if DEBUG: raw_input("Check the summary before going ahead!")

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                     max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                         max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=10)

        model.fit_generator(training_generator,
                            epochs=n.EPOCHS,
                            workers=3,
                            use_multiprocessing=True,
                            callbacks=[checkpointer])
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') ])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))

def bidirectional_dense(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                            _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None):
    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index:
        split_point = index + 1
    else:
        split_point = lambda x: int(len(x) * .70)

    def train_split(x):
        return x[:split_point(x)]

    def test_split(x):
        if _index:
            return x[split_point(x):]
        return x[split_point(x):int(.80 * len(x))]

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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
        nr_hidden = 256

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        # encode = n._BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5)
        encode = n._simple_BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5, return_sequences=False)
        dense = n._simpleDense(max_length, nr_hidden)
        simpler_dense = n._simpleDense_with_units(1, nr_hidden/2)
        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            ques_dense = dense(ques_encoded)
            path_dense = dense(path_encoded)

            concat = n.concatenate([ques_dense,path_dense])

            dot_score = simpler_dense(concat)
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])
        output = Lambda(lambda x: n.concatenate([x[0], x[1], x[2]], axis=-1))([pos_score, neg_score,loss])

        # output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())
        # if DEBUG: raw_input("Check the summary before going ahead!")

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                     max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                         max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=10)

        model.fit_generator(training_generator,
                            epochs=n.EPOCHS,
                            workers=3,
                            use_multiprocessing=True,
                            callbacks=[checkpointer])
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') ])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))

def parikh(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train=10,
           _neg_paths_per_epoch_test=1000, _index=None, _transfer_model_path=None):

    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j+10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]

    # Shuffle these matrices together @TODO this!
    np.random.seed(0) # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index:
        split_point = index + 1
    else:
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

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions)*neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions)*(neg_paths_per_epoch_test+1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = n._BiRNNEncoding(max_length, embedding_dims,  nr_hidden, 0.5)
        # encode = n._simple_BiRNNEncoding(max_length, embedding_dims,  nr_hidden, 0.5)
        # encode = LSTM(max_length)(encode)
        attend = n._Attention(max_length, nr_hidden, dropout=0.6, L2=0.01)
        align = n._SoftAlignment(max_length, nr_hidden)
        compare = n._Comparison(max_length, nr_hidden, dropout=0.6, L2=0.01)
        entail = n._Entailment(nr_hidden, 1, dropout=0.4, L2=0.01)
        dense = n._simpleDense(max_length*2,nr_hidden*2)
        # encode_step_2 = n._simple_CNNEncoding(max_length*2, embedding_dims, nr_hidden, 0.5)


        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # ques_encoded_last_output = ques_encoded[:,-1,:]
            ques_encoded_last_output = Lambda(lambda x: x[:,-1,:])(ques_encoded)
            # path_encoded_last_output = path_encoded[:,-1,:]
            path_encoded_last_output = Lambda(lambda x: x[:,-1,:])(path_encoded)

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


			# poop
            ques_concat = n.concatenate(
                [feats_ques,ques_encoded_last_output], axis=-1
            )

            # ques_concat = n.merge(
            #     [feats_ques, ques_encoded_last_output]
            # )
			#
            path_concat = n.concatenate(
                [feats_path,path_encoded_last_output], axis=-1
            )

            # path_concat = n.merge(
            #     [feats_path, path_encoded_last_output]
            # )

            dense_ques = dense(ques_concat)
            dense_path = dense(path_concat)

            # new_ques = encode_step_2(feats_ques)
            # new_path = encode_step_2(feats_path)
            dot_score = n.dot([dense_ques, dense_path], axes=-1, normalize=True)
            # dot_score = n.dot([feats_ques, feats_path], axes=-1, normalize=True)
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])


        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path], outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER, loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=CHECK_VALIDATION_ACC_PERIOD)

        model.fit_generator(training_generator, epochs=n.EPOCHS, workers=3, use_multiprocessing=True, callbacks=[checkpointer])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))


def maheshwari(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train=10,
               _neg_paths_per_epoch_test=1000, _index=None, _transfer_model_path=None):

    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j+10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]

    # Shuffle these matrices together @TODO this!
    np.random.seed(0) # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index:
        split_point = index + 1
    else:
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

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions)*neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions)*(neg_paths_per_epoch_test+1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = n._BiRNNEncoding(max_length, embedding_dims,  nr_hidden, 0.5)
        # encode = n._simple_BiRNNEncoding(max_length, embedding_dims,  nr_hidden, 0.5)
        # encode = LSTM(max_length)(encode)
        attend = n._Attention(max_length, nr_hidden, dropout=0.6, L2=0.01)
        align = n._SoftAlignment(max_length, nr_hidden)
        compare = n._Comparison(max_length, nr_hidden, dropout=0.6, L2=0.01)
        entail = n._Entailment(nr_hidden, 1, dropout=0.4, L2=0.01)

        x_ques_embedded = embed(x_ques)
        x_pos_path_embedded = embed(x_pos_path)
        x_neg_path_embedded = embed(x_neg_path)

        ques_encoded = encode(x_ques_embedded)
        pos_path_encoded = encode(x_pos_path_embedded)
        neg_path_encoded = encode(x_neg_path_embedded)

        def getScore(path_pos, path_neg):

            attention = attend(path_pos, path_neg)

            align_pos = align(path_pos, attention)
            align_neg = align(path_neg, attention, transpose=True)

            feats_pos = compare(path_pos, align_pos)
            feats_neg = compare(path_neg, align_neg)

            return feats_pos, feats_neg

        pos_path_attended, neg_path_attended = getScore(pos_path_encoded, neg_path_encoded)

        pos_score = n.dot([ques_encoded, pos_path_attended], axes=-1)
        neg_score = n.dot([ques_encoded, neg_path_attended], axes=-1)

        # neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path], outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER, loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, train_questions, train_pos_paths, train_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=CHECK_VALIDATION_ACC_PERIOD)

        model.fit_generator(training_generator, epochs=n.EPOCHS, workers=3, use_multiprocessing=True, callbacks=[checkpointer])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))


def parikh_dot(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train=10,
               _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None):
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index:
        split_point = index + 1
    else:
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

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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
        nr_hidden = 64

        # holographic_forward = Dense(1, activation='sigmoid')
        # final_forward = Dense(1, activation='sigmoid')

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = n._BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5)
        encode_simple = n._simple_BiRNNEncoding(max_length, embedding_dims, nr_hidden/2, 0.4)
        # encode = n._simple_BiRNNEncoding(max_length, embedding_dims,  nr_hidden, 0.5)
        # encode = LSTM(max_length)(encode)
        attend = n._Attention(max_length, nr_hidden, dropout=0.6, L2=0.01)
        align = n._SoftAlignment(max_length, nr_hidden)
        compare = n._Comparison(max_length, nr_hidden, dropout=0.6, L2=0.01)
        entail = n._Entailment(nr_hidden, 1, dropout=0.4, L2=0.01)
        dense = n._simpleDense(max_length , int(nr_hidden/2))

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            ques_encoded_dot = encode_simple(x_ques_embedded)
            path_encoded_dot = encode_simple(x_path_embedded)

            # ques_encoded_last_output = ques_encoded[:,-1,:]
            # ques_encoded_last_output = Lambda(lambda x: x[:,-1,:])(ques_encoded)
            # # path_encoded_last_output = path_encoded[:,-1,:]
            # path_encoded_last_output = Lambda(lambda x: x[:,-1,:])(path_encoded)

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


            #
            # ques_concat = n.concatenate(
            #     [feats_ques,ques_encoded_dot], axis=-1
            # )

            # print ques_concat.shape

            ques_concat = n.merge(
                [feats_ques, ques_encoded_dot]
            )
            #
            # path_concat = n.concatenate(
            #     [feats_path,path_encoded_dot], axis=-1
            # )
			#
            # print path_concat.shape

            path_concat = n.merge(
                [feats_path, path_encoded_dot]
            )

            # dense_ques = dense(ques_concat)
            # dense_path = dense(path_concat)

            # new_ques = encode_step_2(feats_ques)
            # new_path = encode_step_2(feats_path)
            dot_score = n.dot([ques_concat, path_concat], axes=-1, normalize=True)
            # dot_score = n.dot([feats_ques, feats_path], axes=-1, normalize=True)
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])


        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path], outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER, loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=CHECK_VALIDATION_ACC_PERIOD)

        model.fit_generator(training_generator, epochs=n.EPOCHS, workers=3, use_multiprocessing=True, callbacks=[checkpointer])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))


def cnn_dot(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                      _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None,dense=False):
    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index:
        split_point = index + 1
    else:
        split_point = lambda x: int(len(x) * .70)

    def train_split(x):
        return x[:split_point(x)]

    def test_split(x):
        if _index:
            return x[split_point(x):]
        else:
            return x[split_point(x):int(.80 * len(x))]

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = n._simpler_CNNEncoding(max_length, embedding_dims, nr_hidden, 0.5,dense=dense)
        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # concatques = n.concatenate([ques_encoded_a,ques_encoded_b,ques_encoded_c],axis=1)
            # concatpath = n.concatenate([path_encoded_a,path_encoded_b,path_encoded_c],axis=1)
            #
            # flatten_ques = Lambda(lambda x: K.flatten(x))([concatques])
            # flatten_path = Lambda(lambda x: K.flatten(x))([concatpath])
            #
            # dense_ques = simpler_dense_ques(flatten_ques)
            # dense_paths = simpler_dense_ques(flatten_paths)


            # flattenques  = flatten_ques([concatques])
            # flattenpath  = flatten_paths([concatpath])
            # holographic_score = holographic_forward(Lambda(lambda x: cross_correlation(x)) ([ques_encoded, path_encoded]))
            dot_score = n.dot([ques_encoded, path_encoded], axes=-1)
            # l1_score = Lambda(lambda x: K.abs(x[0]-x[1]))([ques_encoded, path_encoded])

            # return final_forward(concatenate([holographic_score, dot_score, l1_score], axis=-1))
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                     max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                         max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=10)

        model.fit_generator(training_generator,
                            epochs=n.EPOCHS,
                            workers=3,
                            use_multiprocessing=True,
                            callbacks=[checkpointer])
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') ])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))


def cnn_dense_dense(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                      _neg_paths_per_epoch_test = 1000, _index=None, _transfer_model_path=None,dense=True):
    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
    if counter > 0:
        print(counter)
        warnings.warn("critical condition needs to be entered")
    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    if _index:
        split_point = index + 1
    else:
        split_point = lambda x: int(len(x) * .70)

    def train_split(x):
        return x[:split_point(x)]

    def test_split(x):
        if _index:
            return x[split_point(x):]
        else:
            return x[split_point(x):int(.80 * len(x))]

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    neg_paths_per_epoch_train = _neg_paths_per_epoch_train
    neg_paths_per_epoch_test = _neg_paths_per_epoch_test
    dummy_y_train = np.zeros(len(train_questions) * neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions) * (neg_paths_per_epoch_test + 1))

    print(train_questions.shape)
    print(train_pos_paths.shape)
    print(train_neg_paths.shape)

    print(test_questions.shape)
    print(test_pos_paths.shape)
    print(test_neg_paths.shape)

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

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = n._simpler_CNNEncoding(max_length, embedding_dims, nr_hidden, 0.5,dense=dense)
        simpler_dense = n._simpleDense_with_units(1, nr_hidden)
        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # concatques = n.concatenate([ques_encoded_a,ques_encoded_b,ques_encoded_c],axis=1)
            # concatpath = n.concatenate([path_encoded_a,path_encoded_b,path_encoded_c],axis=1)
            #
            # flatten_ques = Lambda(lambda x: K.flatten(x))([concatques])
            # flatten_path = Lambda(lambda x: K.flatten(x))([concatpath])
            #
            # dense_ques = simpler_dense_ques(flatten_ques)
            # dense_paths = simpler_dense_ques(flatten_paths)


            # flattenques  = flatten_ques([concatques])
            # flattenpath  = flatten_paths([concatpath])
            # holographic_score = holographic_forward(Lambda(lambda x: cross_correlation(x)) ([ques_encoded, path_encoded]))
            concat = n.concatenate([ques_encoded, path_encoded])
            dot_score = simpler_dense(concat)
            # dot_score = n.dot([ques_encoded, path_encoded], axes=-1)
            # l1_score = Lambda(lambda x: K.abs(x[0]-x[1]))([ques_encoded, path_encoded])

            # return final_forward(concatenate([holographic_score, dot_score, l1_score], axis=-1))
            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])
        output = Lambda(lambda x: n.concatenate([x[0], x[1], x[2]], axis=-1))([pos_score, neg_score, loss])
        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

        """
            Check if we intend to transfer weights from any other model.
        """
        if _transfer_model_path:
            model = n.load_pretrained_weights(_new_model=model, _trained_model_path=_transfer_model_path)

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]

        training_generator = n.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                     max_length, neg_paths_per_epoch_train, n.BATCH_SIZE)
        validation_generator = n.ValidationDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                                         max_length, neg_paths_per_epoch_test, 9999)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_pos_paths, test_neg_paths,
                                               monitor='val_metric',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max',
                                               period=10)

        model.fit_generator(training_generator,
                            epochs=n.EPOCHS,
                            workers=3,
                            use_multiprocessing=True,
                            callbacks=[checkpointer])
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') ])

        print("Precision (hits@1) = ",
              n.rank_precision(model, test_questions, test_pos_paths, test_neg_paths, 1000, 10000))



if __name__ == "__main__":

    # Parse arguments
    GPU = sys.argv[1].strip().lower()
    model = sys.argv[2].strip().lower()
    DATASET = sys.argv[3].strip().lower()
    TRANSFER_MODEL_PATH = None

    # See if the args are valid.
    while True:
        try:
            assert GPU in ['0', '1', '2', '3']
            assert model in ['birnn_dot', 'parikh', 'birnn_dense_dot', 'maheshwari', 'birnn_dense_sigmoid','cnn','cnn_dense_dot',
                             'cnn_dense_dense','parikh_dot','birnn_dot_qald', 'two_birnn_dot','birnn_dense','birnn_dot_pointwise']
            assert DATASET in ['lcquad', 'qald', 'transfer-a', 'transfer-b', 'transfer-c', 'transfer-proper-qald']
            break
        except AssertionError:
            GPU = raw_input("Did not understand which gpu to use. Please write it again: ")
            model = raw_input("Did not understand which model to use. Please write it again: ")
            DATASET = raw_input("Did not understand which dataset to use. Please write it again: ")

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    n.MODEL = 'core_chain/'+model
    n.DATASET = DATASET

    # Load relations and the data
    relations = n.load_relation()

    # @TODO: manage transfer-proper

    if DATASET == 'qald':

        id_train = json.load(open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset':DATASET}, "qald_id_big_data_train.json")))
        id_test = json.load(open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset':DATASET}, "qald_id_big_data_test.json")))

        index = len(id_train) - 1
        FILENAME = 'combined_qald.json'

        json.dump(id_train + id_test, open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset':DATASET}, FILENAME), 'w+'))

    elif DATASET == 'lcquad':
        # n.BATCH_SIZE = 1760
        FILENAME, index = "id_big_data.json", None

    elif DATASET == 'transfer-a':
        FILENAME, index = prepare_transfer_learning.transfer_a()

    elif DATASET == 'transfer-b':
        FILENAME, index = prepare_transfer_learning.transfer_b()

    elif DATASET == 'transfer-c':
        FILENAME, index = prepare_transfer_learning.transfer_c()

    elif DATASET == 'transfer-proper-qald':
        """
            Load model trained on LCQuAD train; and is now going to be trained on QALD train. 
        """
        id_train = json.load(open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'}, "qald_id_big_data_train.json")))
        id_test = json.load(open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset':'qald'}, "qald_id_big_data_test.json")))

        index = len(id_train) - 1
        FILENAME = 'combined_qald.json'

        json.dump(id_train + id_test, open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset':DATASET}, FILENAME), 'w+'))
        TRANSFER_MODEL_PATH = os.path.join(n.MODEL_DIR % {'model':n.MODEL, 'dataset':'lcquad'}, LCQUAD_BIRNN_MODEL)

    else:
        warnings.warn("Code never comes here. ")
        FILENAME, index = None, None

    vectors, questions, pos_paths, neg_paths = n.create_dataset_pairwise(FILENAME, n.MAX_SEQ_LENGTH,
                                                                         relations)

    if DEBUG: print("About to choose models")

    # Start training
    if model == 'birnn_dot':

        print("About to run BiDirectionalRNN with Dot")
        bidirectional_dot(GPU, vectors, questions, pos_paths, neg_paths, 100, 1000, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'two_birnn_dot':

        print("About to run BiDirectionalRNN with Dot")
        two_bidirectional_dot(GPU, vectors, questions, pos_paths, neg_paths, 100, 1000, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'birnn_dot_sigmoid':

        print("About to run BiDirectionalRNN with Dot and Sigmoid loss")
        bidirectional_dot_sigmoidloss(GPU, vectors, questions, pos_paths, neg_paths, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'birnn_dense_dot':

        print("About to run BiDirectionalRNN with Dense")
        bidirectional_dense_dot(GPU, vectors, questions, pos_paths, neg_paths, 10, 1000, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'birnn_dense':

        print("About to run BiDirectionalRNN with Dense")
        bidirectional_dense(GPU, vectors, questions, pos_paths, neg_paths, 10, 1000, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'parikh':

        print("About to run Parikh et al model")
        parikh(GPU, vectors, questions, pos_paths, neg_paths, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'maheshwari':

        print("About to run Maheshwari et al model")
        maheshwari(GPU, vectors, questions, pos_paths, neg_paths, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'cnn':

        print("About to run cnn et al model")
        cnn_dot(GPU, vectors, questions, pos_paths, neg_paths, 100, 1000, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'parikh_dot':

        print("About to run cnn et al model")
        parikh_dot(GPU, vectors, questions, pos_paths, neg_paths, index, _transfer_model_path=TRANSFER_MODEL_PATH)
    elif model == 'birnn_dot_pointwise':

        print("About to run birnn_dot_pointwise ")
        bidirectional_dot_pointwise(GPU, vectors, questions, pos_paths, neg_paths, 100, 1000, index, _transfer_model_path=TRANSFER_MODEL_PATH)

    elif model == 'cnn_dense_dot':

        print("About to run cnn_dense_dot ")
        cnn_dot(GPU, vectors, questions, pos_paths, neg_paths, 100, 1000, index, _transfer_model_path=TRANSFER_MODEL_PATH,dense=True)

    elif model == 'cnn_dense_dense':

        print("About to run cnn_dense_dense ")
        cnn_dense_dense(GPU, vectors, questions, pos_paths, neg_paths, 100, 1000, index, _transfer_model_path=TRANSFER_MODEL_PATH,dense=True)
    else:
        warnings.warn("Did not choose any model.")
        if DEBUG:
            print("sysargs are: ", GPU, model)


