"""
    The file with different models for corechain ranking.
    bidirectional_dot

"""

from __future__ import absolute_import
import os
import sys
import warnings
import numpy as np
from macpath import norm_error

import keras.backend.tensorflow_backend as K
from keras.models import Model
from keras.layers import Input, Concatenate, Lambda

import network as n

# Macros
DEBUG = True
LCQUAD = True
DATA_DIR_CORECHAIN = './data/models/core_chain/%(model)s/lcquad/' if LCQUAD else './data/models/%(model)s/core_chain/qald/'
RES_DIR_PAIRWISE_CORECHAIN = './data/data/core_chain_pairwise/lcquad/' if LCQUAD else './data/data/core_chain_pairwise/qald/'
RES_DIR_POINTWISE_CORECHAIN = './data/data/core_chain_pointwise/lcquad/' if LCQUAD else './data/data/core_chain_pointwise/qald/'
CHECK_VALIDATION_ACC_PERIOD = 10


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def bidirectional_dot_sigmoidloss(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                                  _neg_paths_per_epoch_test = 1000):
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


def bidirectional_dot(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10,
                      _neg_paths_per_epoch_test = 1000):
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

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
                      outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.custom_loss)

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
                      _neg_paths_per_epoch_test = 1000):
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
        encode = n._simple_BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5, return_sequences=False)
        dense = n._simpleDense(max_length, nr_hidden)

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            # ques_encoded = K.reshape(ques_encoded, (256*max_length))
            # path_encoded = K.reshape(path_encoded, (256*max_length))

            # print(ques_encoded.shape, path_encoded.shape)
            # raw_input("Check shape")

            # holographic_score = holographic_forward(Lambda(lambda x: cross_correlation(x)) ([ques_encoded, path_encoded]))
            # dot_score = n.dot([ques_encoded, path_encoded], axes=-1)
            # concatenated_vector = Concatenate([ques_encoded, path_encoded])
            dense_score = dense(ques_encoded, path_encoded)
            # l1_score = Lambda(lambda x: K.abs(x[0]-x[1]))([ques_encoded, path_encoded])

            # return final_forward(concatenate([holographic_score, dot_score, l1_score], axis=-1))
            return dense_score

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


def parikh(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10, _neg_paths_per_epoch_test = 1000):

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

            dot_score = n.dot([feats_ques, feats_path], axes=-1)

            return dot_score

        pos_score = getScore(x_ques, x_pos_path)
        neg_score = getScore(x_ques, x_neg_path)

        loss = Lambda(lambda x: K.maximum(0., 1.0 - x[0] + x[1]))([pos_score, neg_score])

        output = n.concatenate([pos_score, neg_score, loss], axis=-1)

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path], outputs=[output])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER, loss=n.custom_loss)

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


def maheshwari(_gpu, vectors, questions, pos_paths, neg_paths, _neg_paths_per_epoch_train = 10, _neg_paths_per_epoch_test = 1000):

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


def pointwise_bidirectional_dot(_gpu, vectors, questions, paths, labels):
    """
        Data Time!
    """
    # Pull the data up from disk
    gpu = _gpu
    max_length = n.MAX_SEQ_LENGTH

    np.random.seed(0)  # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    split_point = lambda x: int(len(x) * .80)

    def train_split(x):
        return x[:split_point(x)]

    def test_split(x):
        return x[split_point(x):]

    train_paths = train_split(paths)
    train_labels = train_split(labels)
    train_questions = train_split(questions)

    test_paths = test_split(paths)
    test_labels = test_split(labels)
    test_questions = test_split(questions)

    print(train_questions.shape)
    print(train_paths.shape)
    print(train_labels.shape)

    print(test_questions.shape)
    print(test_paths.shape)
    print(test_labels.shape)

    with K.tf.device('/gpu:' + gpu):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True)))
        """
            Model Time!
        """
        max_length = train_questions.shape[1]
        # Define input to the models
        x_ques = Input(shape=(max_length,), dtype='int32', name='x_ques')
        x_path = Input(shape=(max_length,), dtype='int32', name='x_path')
        # y = Input(shape=(1,), dtype='int32', name='y')

        embedding_dims = vectors.shape[1]
        nr_hidden = 128

        embed = n._StaticEmbedding(vectors, max_length, embedding_dims, dropout=0.2)
        encode = n._simple_BiRNNEncoding(max_length, embedding_dims, nr_hidden, 0.5)

        def getScore(ques, path):
            x_ques_embedded = embed(ques)
            x_path_embedded = embed(path)

            ques_encoded = encode(x_ques_embedded)
            path_encoded = encode(x_path_embedded)

            dot_score = n.dot([ques_encoded, path_encoded], axes=-1)

            return dot_score

        dotscore = getScore(x_ques, x_path)

        # Model time!
        model = Model(inputs=[x_ques, x_path],
                      outputs=[dotscore])

        print(model.summary())

        model.compile(optimizer=n.OPTIMIZER,
                      loss=n.LOSS)

        # Prepare training data
        training_generator = n.PointWiseTrainingDataGenerator(train_questions, train_paths, train_labels,
                                                     max_length, n.BATCH_SIZE)

        # smart_save_model(model)
        json_desc, dir = n.get_smart_save_path(model)
        model_save_path = os.path.join(dir, 'model.h5')

        checkpointer = n.CustomModelCheckpoint(model_save_path, test_questions, test_paths, test_labels,
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
              n.rank_precision(model, test_questions, test_paths, test_labels, 1000, 10000))

if __name__ == "__main__":
    gpu = sys.argv[1].strip().lower()
    model = sys.argv[2].strip().lower()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # See if the args are valid.
    while True:
        try:
            assert gpu in ['0','1','2','3']
            assert model.lower() in ['birnn_dot', 'parikh', 'birnn_dense', 'maheshwari', 'birnn_dense_sigmoid', 'pointwise_birnn_dot']
            break
        except AssertionError:
            gpu = raw_input("Did not understand which gpu to use. Please write it again: ")
            model = raw_input("Did not understand which model to use. Please write it again: ")

    if DEBUG: print("About to choose models")

    # Start training
    if model == 'birnn_dot':
        # Specify the directory to save model
        n.MODEL_DIR = DATA_DIR_CORECHAIN % {"model": model}  # **CHANGE WHEN CHANGING MODEL!**
        n.CACHE_DATA_PAIRWISE_DIR = RES_DIR_PAIRWISE_CORECHAIN

        # Load relations and the data
        relations = n.load_relation('resources_v8/relations.pickle')
        vectors, questions, pos_paths, neg_paths = n.create_dataset_pairwise("id_big_data.json", n.MAX_SEQ_LENGTH,
                                                                             relations)
        print("About to run BiDirectionalRNN with Dot")
        bidirectional_dot(gpu, vectors, questions, pos_paths, neg_paths, 100, 1000)

    elif model == 'birnn_dot_sigmoid':
        # Specify the directory to save model
        n.MODEL_DIR = DATA_DIR_CORECHAIN % {"model": model}  # **CHANGE WHEN CHANGING MODEL!**
        n.CACHE_DATA_PAIRWISE_DIR = RES_DIR_PAIRWISE_CORECHAIN

        # Load relations and the data
        relations = n.load_relation('resources_v8/relations.pickle')
        vectors, questions, pos_paths, neg_paths = n.create_dataset_pairwise("id_big_data.json", n.MAX_SEQ_LENGTH,
                                                                             relations)

        print("About to run BiDirectionalRNN with Dot and Sigmoid loss")
        bidirectional_dot_sigmoidloss(gpu, vectors, questions, pos_paths, neg_paths)

    if model == 'birnn_dense':
        # Specify the directory to save model
        n.MODEL_DIR = DATA_DIR_CORECHAIN % {"model": model}  # **CHANGE WHEN CHANGING MODEL!**
        n.CACHE_DATA_PAIRWISE_DIR = RES_DIR_PAIRWISE_CORECHAIN

        # Load relations and the data
        relations = n.load_relation('resources_v8/relations.pickle')
        vectors, questions, pos_paths, neg_paths = n.create_dataset_pairwise("id_big_data.json", n.MAX_SEQ_LENGTH,
                                                                             relations)

        print("About to run BiDirectionalRNN with Dense")
        bidirectional_dense(gpu, vectors, questions, pos_paths, neg_paths, 10, 1000)

    elif model == 'parikh':
        # Specify the directory to save model
        n.MODEL_DIR = DATA_DIR_CORECHAIN % {"model": model}  # **CHANGE WHEN CHANGING MODEL!**
        n.CACHE_DATA_PAIRWISE_DIR = RES_DIR_PAIRWISE_CORECHAIN

        # Load relations and the data
        relations = n.load_relation('resources_v8/relations.pickle')
        vectors, questions, pos_paths, neg_paths = n.create_dataset_pairwise("id_big_data.json", n.MAX_SEQ_LENGTH,
                                                                             relations)

        print("About to run Parikh et al model")
        parikh(gpu, vectors, questions, pos_paths, neg_paths)
    elif model == 'maheshwari':
        # Specify the directory to save model
        n.MODEL_DIR = DATA_DIR_CORECHAIN % {"model": model}  # **CHANGE WHEN CHANGING MODEL!**
        n.CACHE_DATA_PAIRWISE_DIR = RES_DIR_PAIRWISE_CORECHAIN

        # Load relations and the data
        relations = n.load_relation('resources_v8/relations.pickle')
        vectors, questions, pos_paths, neg_paths = n.create_dataset_pairwise("id_big_data.json", n.MAX_SEQ_LENGTH,
                                                                             relations)

        print("About to run Maheshwari et al model")
        maheshwari(gpu, vectors, questions, pos_paths, neg_paths)

    # #######################
    # Pointwise models hereon
    # #######################
    elif model == 'pointwise_birnn_dot':
        # Specify the directory to save model
        n.MODEL_DIR = DATA_DIR_CORECHAIN % {"model": model}  # **CHANGE WHEN CHANGING MODEL!**
        n.CACHE_DATA_POINTWISE_DIR = RES_DIR_POINTWISE_CORECHAIN

        # Load relations and the data
        relations = n.load_relation('resources_v8/relations.pickle')
        vectors, questions, paths, labels = n.create_dataset_pointwise("id_big_data.json", n.MAX_SEQ_LENGTH,
                                                                             relations)

        print("About to run Pointwise BidirectionalRNN with a dot at the helm. God save the queen.")
        pointwise_bidirectional_dot(gpu, vectors, questions, paths, labels)
        
    else:
        warnings.warn("Did not choose any model.")
        if DEBUG:
            print("sysargs are: ", gpu, model)
