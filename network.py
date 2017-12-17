import keras,math
import cPickle as pickle
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model

from keras import backend as K
import os

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("tensorflow")


#data loading from the secondary memory
FILE_PATH = 'data/training/100'
paths = pickle.load(open(FILE_PATH + '/P'))
questions = pickle.load(open(FILE_PATH + '/Q'))
true_paths = pickle.load(open(FILE_PATH + '/Y'))
#
x_path_train = np.asarray(paths[:int(len(paths)*.80)]).astype('float32')
y_train = np.asarray(true_paths[:int(len(true_paths)*.80)]).astype('float32')
x_path_test = np.asarray(paths[int(len(paths)*.80):]).astype('float32')
y_test = np.asarray(true_paths[int(len(true_paths)*.80):]).astype('float32')
q_path_train = np.asarray(questions[:int(len(questions)*.80)]).astype('float32')
q_path_test = np.asarray(questions[int(len(questions)*.80):]).astype('float32')
print(x_path_train.shape[0], 'train samples')

# x_path_train = np.random.rand((800,24,300))
# q_path_train = np.random.rand((800,24,300))
# x_path_test = np.random.rand((200,24,300))
# q_path_test = np.random.rand((200,24,300))
# y_train = np.random.rand((800,1))
# y_test = np.random.rand((200,1))

path_input_shape = x_path_train.shape[1:]
question_input_shape = q_path_train.shape[1:]

# path_input_shape = x_path_train.shape
# question_input_shape = q_path_train.shape
# Declaring input tensors
x_ques = Input(shape=question_input_shape)
x_path = Input(shape=path_input_shape)
# neg_path = Input(shape=(20, 300))


'''
    Encoder
'''
# Create two different encoders
path_encoder = LSTM(64)
ques_encoder = LSTM(64)

# Encode the question
encoded_question = ques_encoder(x_ques)

# Encode the pos and neg path using the same path encoder
encoded_path = path_encoder(x_path)
# encoded_neg_path = path_encoder(neg_path)

# Concatenate question with the two paths
merged_path_ques = keras.layers.concatenate([encoded_question, encoded_path], axis=-1)
# merged_negpath_ques = keras.layers.concatenate([encoded_question, encoded_pos_path], axis=-1)

'''
    Classifier
'''
# Declare a dense layer
classifier = Dense(1, activation='sigmoid')

# Use it on the pos and neg examples
y = classifier(merged_path_ques)
# y_neg = classifier(merged_negpath_ques)

model = Model(inputs=[ques_encoder, path_encoder], outputs=y)

model.compile(optimizer=["sgd"],
              loss="binary_crossentropy",
              metrics=["precision", "recall", "accuracy"])
model.fit([x_path_train,q_path_train],y_train,batch_size=1,epochs=100)