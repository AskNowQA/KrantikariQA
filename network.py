# Shared Feature Extraction Layer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate

import cPickle as pickle
import numpy as np



#data loading from the secondary memory
FILE_PATH = 'data/training/100'
paths = pickle.load(open(FILE_PATH + '/P'))
questions = pickle.load(open(FILE_PATH + '/Q'))
true_paths = pickle.load(open(FILE_PATH + '/Y'))


x_path_train = np.asarray(paths[:int(len(paths)*.80)]).astype('float32')
y_train = np.asarray(true_paths[:int(len(true_paths)*.80)]).astype('float32')
x_path_test = np.asarray(paths[int(len(paths)*.80):]).astype('float32')
y_test = np.asarray(true_paths[int(len(true_paths)*.80):]).astype('float32')
q_path_train = np.asarray(questions[:int(len(questions)*.80)]).astype('float32')
q_path_test = np.asarray(questions[int(len(questions)*.80):]).astype('float32')
print(x_path_train.shape[0], 'train samples')


path_input_shape = x_path_train.shape[1:]
question_input_shape = q_path_train.shape[1:]


# Encode the pos and neg path using the same path encoder and also the question
x_ques = Input(shape=path_input_shape)
ques_encoder = LSTM(64)(x_ques)
x_path = Input(shape=question_input_shape)
path_encoder = LSTM(64)(x_path)

# Concatenate question with the two paths
merge = concatenate([ques_encoder, path_encoder])

output = Dense(1, activation='sigmoid')(merge)
model = Model(inputs=[x_ques,x_path], outputs=output)
print(model.summary())
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([x_path_train,q_path_train],y_train,batch_size=1,epochs=100)