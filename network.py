# Shared Feature Extraction Layer
import os
import json
import numpy as np
import keras.backend.tensorflow_backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation, RepeatVector, Reshape, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate, dot, subtract, maximum
from keras.activations import softmax
from keras import optimizers, metrics

gpu = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# Some Macros
DEBUG = True
DATA_DIR = './data/training/pairwise'
EPOCHS = 200
BATCH_SIZE = 2000 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
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


'''
    More Helper Functions
'''
def smart_save_model(model):
    """
        Function to properly save the model to disk.
            If the model config is the same as one already on disk, overwrite it.
            Else make a new folder and write things there

    :return: None
    """

    # Get the model description
    desc = model.to_json()

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
        if json.load(open(os.path.join(l_dir, 'model.json'))) == desc:
            # Same desc. Just save stuff here
            if DEBUG:
                print "network.py:smart_save_model: Saving model in %s" % l_dir
            model.save(os.path.join(l_dir, 'model.h5'))

        else:
            # Diff model. Make new folder and do stuff. @TODO this
            new_num = int(dir_nums[-1]) + 1
            if new_num < 10:
                new_num = str('0') + str(new_num)
            else:
                new_num = str(new_num)

            l_dir = os.path.join(DATA_DIR, "model_" + new_num)
            os.mkdir(l_dir)
            raise IOError

    except IOError:

        # Apparently there's nothing here. Let's set camp.
        if DEBUG:
            print "network.py:smart_save_model: Saving model in %s" % l_dir
        model.save(os.path.join(l_dir, 'model.h5'))
        json.dump(desc, open(os.path.join(l_dir, 'model.json'), 'w+'))


def zeroloss(yt, yp):
    return 0.0


def custom_loss(y_true, y_pred):
    '''
        max margin loss
    '''
    # y_pos = y_pred[0]
    # y_neg= y_pred[1]
    return K.sum(K.maximum(1.0 - y_pred, 0.))


if __name__ == "__main__":
    """
        Data Time!
    """
    # Pull the data up from disk
    pos_paths = np.load(DATA_DIR + '/tP.npz')
    neg_paths = np.load(DATA_DIR + '/fP.npz')
    questions = np.load(DATA_DIR + '/Q.npz')

    # Shuffle these matrices together @TODO this!
    np.random.seed(0) # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    split_point = lambda x: int(len(x)/20 * .80) * 20

    def train_split(x):
        return np.asarray(x[:split_point(x)]).astype('float32')
    def test_split(x):
        return np.asarray(x[split_point(x):]).astype('float32')

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    # indices = np.random.permutation(train_pos_paths.shape[0])

    # train_pos_paths = train_pos_paths[indices]
    # train_neg_paths = train_neg_paths[indices]
    # train_questions = train_questions[indices]

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    question_input_shape = train_questions.shape[1:]
    path_input_shape = train_pos_paths.shape[1:]

    with K.tf.device('/gpu:' + gpu):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True)))
        """
            Model Time!
        """
        # Define input to the models
        x_ques = Input(shape=question_input_shape)
        x_pos_path = Input(shape=path_input_shape)
        x_neg_path = Input(shape=path_input_shape)

        # Encode the questions
        ques_encoder = Bidirectional(LSTM(64, dropout = 0.3))
        # forwardVector = Dense(32)
        # dropout = Dropout(0.5)
        # ques_encoded = (forwardVector(ques_encoder(x_ques)))
        ques_encoded = ques_encoder(x_ques)

        # Encode 21 paths
        path_encoder = Bidirectional(LSTM(64, dropout = 0.3))
        # forwardVector = Dense(32)
        # dropout = Dropout(0.5)
        # pos_path_encoded = (forwardVector(path_encoder(x_pos_path)))
        # neg_path_encoded = (forwardVector(path_encoder(x_neg_path)))
        pos_path_encoded = path_encoder(x_pos_path)
        neg_path_encoded = path_encoder(x_neg_path)

        # forwardMatrix = Dense(64, activation='tanh')
        # dropout = Dropout(0.5)
        # forwardVector = Dense(1, activation='sigmoid')

        pos_score = (dot([ques_encoded, pos_path_encoded], axes=-1))
        neg_score = (dot([ques_encoded, neg_path_encoded], axes=-1))

        # Model time!
        model = Model(inputs=[x_ques, x_pos_path, x_neg_path],
            outputs=[pos_score, subtract([pos_score, neg_score])])

        print(model.summary())

        model.compile(optimizer=OPTIMIZER,
                      loss=[None, custom_loss], loss_weights=[None, 1.])

        # Prepare training data
        training_input = [train_questions, train_pos_paths, train_neg_paths]
        dummy_y = np.zeros(train_questions.shape[0])
        model.fit(training_input, dummy_y, batch_size=BATCH_SIZE, epochs=EPOCHS)

        smart_save_model(model)

        # Prepare test data

        only_questions = test_questions[range(0, test_questions.shape[0], 20)]
        only_pos_paths = test_pos_paths[range(0, test_pos_paths.shape[0], 20)]

        pos_outputs = np.array(model.predict([only_questions, only_pos_paths, only_pos_paths])[0])
        neg_outputs = np.array(model.predict([test_questions, test_neg_paths, test_neg_paths])[0])
        neg_outputs = np.reshape(neg_outputs, [only_pos_paths.shape[0], 20])
        all_outputs = np.hstack([pos_outputs, neg_outputs])

        precision = float(len(np.where(np.argmax(all_outputs, axis=1)==0)[0]))/all_outputs.shape[0]
        print "Precision (hits@1) = ", precision

# print "Evaluation Complete"
# print "Loss     = ", results[0]
# print "F1 Score = ", results[1]
# print "Accuracy = ", results[2]
