# Shared Feature Extraction Layer
import os
import json
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate, dot
from keras.activations import softmax
from keras import optimizers, metrics


# Some Macros
DEBUG = True
DATA_DIR = './data/training/full'
EPOCHS = 200
BATCH_SIZE = 200 # Around 11 splits for full training dataset
LEARNING_RATE = 0.002
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


def custom_loss(y_true, y_pred):
    '''
        max margin loss
    '''
    # y_neg = K.concatenate(y_pred[1:])
    # y_pos = K.repeat(y_pred, 20)
    # K.eval(y_pos)
    # K.eval(y_neg)
    y_pos = y_pred[0]
    y_neg= y_pred[1]
    return K.mean(K.maximum(1. - y_pos +  y_neg, 0.) , axis=-1)

"""
    Data Time!
"""
# Pull the data up from disk
x_p = np.load(open(DATA_DIR + '/P.npz'))
x_q = np.load(open(DATA_DIR + '/Q.npz'))
y = np.load(open(DATA_DIR + '/Y.npz'))

# Shuffle these matrices together @TODO this!
np.random.seed(0) # Random train/test splits stay the same between runs
indices = np.random.permutation(x_p.shape[0])
x_p = x_p[indices]
x_q = x_q[indices]
y = y[indices]

# Divide the data into diff blocks
x_path_train = np.asarray(x_p[:int(len(x_p) * .80)]).astype('float32')
y_train = np.asarray(y[:int(len(y) * .80)]).astype('float32')
x_path_test = np.asarray(x_p[int(len(x_p) * .80):]).astype('float32')
y_test = np.asarray(y[int(len(y) * .80):]).astype('float32')
q_path_train = np.asarray(x_q[:int(len(x_q) * .80)]).astype('float32')
q_path_test = np.asarray(x_q[int(len(x_q) * .80):]).astype('float32')

question_input_shape = q_path_train.shape[1:]
path_input_shape = x_path_train.shape[2:]


"""
    Model Time!
"""
# Define input to the models
x_ques = Input(shape=question_input_shape)
x_paths = [ Input(shape=path_input_shape) for x in range(x_path_train.shape[1])]

# Encode the question
ques_encoded = LSTM(64)(x_ques)

# Encode 21 paths
path_encoder = LSTM(64)
path_encoded = [path_encoder(x) for x in x_paths]

# For every path, concatenate question with the path
merges = [dot([ques_encoded, x],axes=-1) for x in path_encoded]
# pos = merges[0]
# neg = merges[1:]


"""
    Run Time
"""
pos = merges[0]
repeat_pos = RepeatVector(20)(pos)
neg = merges[1:]
concat_neg = concatenate(neg)
# Prepare input tensors
inputs = [x_ques] + x_paths

# Model time!
model = Model(inputs=inputs, outputs=[repeat_pos,concat_neg])

print(model.summary())

model.compile(optimizer=OPTIMIZER,
              loss=custom_loss,
              metrics=['accuracy'])

# Prepare training data
training_input = [q_path_train] + [x_path_train[:, i, :, :] for i in range(x_path_train.shape[1])]
model.fit(training_input, [y_train,y_train], batch_size=1, epochs=EPOCHS)


smart_save_model(model)

# Prepare test data
testing_input = [q_path_test] + [x_path_test[:, i, :, :] for i in range(x_path_test.shape[1])]
results = model.evaluate(testing_input, y_test)
print "Evaluation Complete"
print "Loss     = ", results[0]
print "F1 Score = ", results[1]
print "Accuracy = ", results[2]