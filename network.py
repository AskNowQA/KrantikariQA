# Shared Feature Extraction Layer
import os
import json
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras import optimizers, metrics


# Some Macros
DATA_DIR = './data/training/multi_path_mini'
EPOCHS = 50

'''
	F1 measure
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

    # Find the current model dirs in the data dir.
    _, dirs, _ = os.walk(DATA_DIR).next()

    # Get the model description
    desc = model.to_json()

    # Find the latest folder in here
    l_dir = os.path.join(DATA_DIR, dirs[-1])    # @TODO: Replace this with alphanum sort

    # Check if the latest dir has the same model as current
    if __name__ == '__main__':
        try:
            if json.load(os.path.join(l_dir, 'model.json')) == desc:
                # Same desc. Just save stuff here
                model.save(os.path.join(l_dir, 'model.h5'))

            else:
                # Diff model. Make new folder and do stuff. @TODO this
                pass

        except IOError:

            # Apparently there's nothing here. Let's set camp.
            model.save(os.path.join(l_dir, 'model.h5'))
            json.dump(desc, open(os.path.join(l_dir, 'model.json'), 'w+'))

"""
    Data Time!
"""
# Pull the data up from disk
x_p = np.load(open(DATA_DIR + '/P.npz'))
x_q = np.load(open(DATA_DIR + '/Q.npz'))
y = np.load(open(DATA_DIR + '/Y.npz'))

# Divide the data into diff blocks
x_path_train = np.asarray(x_p[:int(len(x_p) * .80)]).astype('float32')
y_train = np.asarray(y[:int(len(y) * .80)]).astype('float32')
x_path_test = np.asarray(x_p[int(len(x_p) * .80):]).astype('float32')
y_test = np.asarray(y[int(len(y) * .80):]).astype('float32')
q_path_train = np.asarray(x_q[:int(len(x_q) * .80)]).astype('float32')
q_path_test = np.asarray(x_q[int(len(x_q) * .80):]).astype('float32')

path_input_shape = x_path_train.shape[1:]
question_input_shape = q_path_train.shape[1:]

"""
    Model Time!
"""
# Encode the pos and neg path using the same path encoder and also the question
x_ques = Input(shape=path_input_shape)
ques_encoder = LSTM(64)(x_ques)
x_path = Input(shape=question_input_shape)
path_encoder = LSTM(64)(x_path)

# Concatenate question with the two paths
merge = concatenate([ques_encoder, path_encoder])

output = Dense(1, activation='sigmoid')(merge)

"""
    Run Time!
"""
# Model time!
model = Model(inputs=[x_ques,x_path], outputs=output)

print(model.summary())

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([x_path_train, q_path_train], y_train, batch_size=1, epochs=EPOCHS)

smart_save_model(model)

