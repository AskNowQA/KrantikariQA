'''
	This file explores various ways of  classifying the number of relationships based on information just available in question
	and not using the knowledge base.
'''

import json
import traceback
import numpy as np
from keras.layers import Embedding, Flatten
from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Reshape, Flatten, Dropout, LSTM, Bidirectional
from keras import regularizers
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

import krantikari as k
from utils import natural_language_utilities as nlutils
from utils import embeddings_interface



LCQUAD_DIR = './resources/data_set.json'
EMBEDDING_DIM = 300



continous_id = {}




def generate_training_data(qald = False,shuffle = True,seed = 42):

	if not qald:
		dataset = json.load(open(LCQUAD_DIR))
	else:
		print "not supported"

	parsed_data = []
	for datapoint in dataset:
		parsed_data_point = k.parse_lcquad(datapoint)
		parsed_data.append(parsed_data_point)

	if shuffle:
		np.random.seed(seed)
		s = np.arange(len(parsed_data))
		np.random.shuffle(s)
		shuffled_data_set = [parsed_data[i] for i in s]
		return shuffled_data_set
	else:
		return parsed_data

def dataset():
	parsed_data = generate_training_data()

	#splitting the data
	train_data = parsed_data[:int(len(parsed_data)*.80)]
	test_data = parsed_data[int(len(parsed_data)*.80):]

	train = []
	for data_point in train_data:
		try:
			train.append([[embeddings_interface.vocabularize(nlutils.tokenize(data_point[u'corrected_question']),_embedding="glove")],[len(data_point[u'path'])]])
		except:
			print traceback.print_exc()
			continue

	test = []
	for data_point in test_data:
		try:
			test.append([[embeddings_interface.vocabularize(nlutils.tokenize(data_point[u'corrected_question']),_embedding="glove")],[len(data_point[u'path'])]])
		except:
			continue

	np.save('resources/train_classifier.npy',train)
	np.save('resources/test_classifier.npy',test)

	#save this as numpy matrix


'''
	Need to add a special end token and also need to pad the sequence.
'''

def max_length(questions):
	max = 0
	for ques in questions:
		if max < ques.shape[0]:
			max = ques.shape[0]
	return max

def get_glove_embeddings():
	from utils.embeddings_interface import __check_prepared__
	__check_prepared__('glove')
	from utils.embeddings_interface import glove_embeddings
	return glove_embeddings


glove_embeddings = get_glove_embeddings()
def load_data():
	train_data = np.load('resources/train_classifier.npy')
	test_data = np.load('resources/test_classifier.npy')

	data = np.vstack((train_data,test_data))
	questions = [i[0][0] for i in data]
	max_sequence_length = max_length(questions)
	questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')

	'''
		A small hack to go around the problem of continous id
	'''
	_temp_id = 0
	for ques in questions:
		for token in ques:
			if token not in continous_id:
				continous_id[token] = _temp_id
				_temp_id = _temp_id + 1
	#Changing the id of the quesiton

	vec_questions = []
	for ques in questions:
		vec_que = [continous_id[i] for i in ques]
		vec_questions.append(vec_que)
	'''
		Still need to convert the glove to id
		vector = glove_embeddings[id]
	'''

	labels = [i[1][0] for i in data]
	labels = to_categorical(labels, num_classes=None)
	x_train, x_test = vec_questions[:int(len(vec_questions)*.80)], vec_questions[int(len(vec_questions)*.80):]
	y_train,y_test = labels[:int(len(vec_questions)*.80)],labels[int(len(vec_questions)*.80):]
	return x_train,y_train,x_test,y_test,max_sequence_length




generate_training_data()
x_train,y_train,x_test,y_test,max_seq_length = load_data()

'''
	Done splitting, padding, embedding
'''


embedding_matrix = np.zeros((len(continous_id) + 1, EMBEDDING_DIM))


for key in continous_id:
	embedding_vector = glove_embeddings[key]
	i = continous_id[key]
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(len(continous_id) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_seq_length,
                            trainable=False)



def cnn_model():
	sequence_input = Input(shape=(max_seq_length,))
	embedded_sequences = embedding_layer(sequence_input)
	x = Conv1D(128, 5, activation='relu',input_shape = (25,300))(embedded_sequences)
	x = MaxPooling1D(2)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	# x = Dropout(0.5)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(2)(x)  # global max pooling
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	preds = Dense(3, activation='softmax')(x)

	model = Model(sequence_input, preds)
	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['acc'])
	model.summary()
	model.fit(np.asarray(x_train), np.asarray(y_train),
			  epochs=30, batch_size=128)
	return model

def rnn_model():
	sequence_input = Input(shape=(max_seq_length,))
	embedded_sequences = embedding_layer(sequence_input)
	x = LSTM(128,dropout=0.3)(embedded_sequences)
	x = Dense(128, activation='relu')(x)
	preds = Dense(3, activation='softmax')(x)
	model = Model(sequence_input, preds)
	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['acc'])
	model.summary()
	model.fit(np.asarray(x_train), np.asarray(y_train),
			  epochs=20, batch_size=128)
	return model


def rnn_cnn_model():
	sequence_input = Input(shape=(max_seq_length,))
	embedded_sequences = embedding_layer(sequence_input)
	x = Conv1D(128, 5, activation='relu',input_shape = (25,300))(embedded_sequences)
	x = MaxPooling1D(2)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(2)(x)  # global max pooling
	# x = Flatten()(x)
	x = LSTM(128)(x)
	x = Dropout(0.5)(x)
	x = Dense(128, activation='relu')(x)
	preds = Dense(3, activation='softmax')(x)

	model = Model(sequence_input, preds)
	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['acc'])
	model.summary()
	model.fit(np.asarray(x_train), np.asarray(y_train),
			  epochs=30, batch_size=128)
	return model



cnn_model = cnn_model()
rnn_model = rnn_model()
rnn_cnn_model = rnn_cnn_model()



cnn_model_predict = cnn_model.predict(x_test)
rnn_model_predict = rnn_model.predict(x_test)
rnn_cnn_model_predict = rnn_cnn_model.predict(x_test)


result = 0
for i in xrange(len(cnn_model_predict)):
	if np.argmax(cnn_model_predict[i]) == np.argmax(y_test[i]) or np.argmax(rnn_model_predict[i]) == np.argmax(y_test[i]):
		result = result + 1

print "combined results are ", result

result = 0
for i in xrange(len(cnn_model_predict)):
	if np.argmax(cnn_model_predict[i]) == np.argmax(y_test[i]):
		result = result + 1
print "cnn model results are ", result


result = 0
for i in xrange(len(rnn_model_predict)):
	if np.argmax(rnn_model_predict[i]) == np.argmax(y_test[i]):
		result = result + 1
print "rnn model results are ", result

result = 0
for i in xrange(len(rnn_cnn_model_predict)):
	if np.argmax(rnn_cnn_model_predict[i]) == np.argmax(y_test[i]):
		result = result + 1
print "rnn model results are ", result



#
#
#
# result = 0
# for i in xrange(len(cnn_model_predict)):
# 	if np.argmax(cnn_model_predict[i]) == np.argmax(y_test[i]):
# 		result = result + 1
#
#
#
# model = rnn_model()
#
# predicted_output = model.predict(x_test)
#
# result = 0
# for i in xrange(len(predicted_output)):
# 	if np.argmax(predicted_output[i]) == np.argmax(y_test[i]):
# 		result = result + 1
#
# print result
# print len(y_test)
#
# stats_dict = {0:0,1:0,2:0}
#
# for i in xrange(len(y_test)):
# 	stats_dict[np.argmax(y_test[i])] = stats_dict[np.argmax(y_test[i])] + 1
#
# stats_dict = {0:0,1:0,2:0}
#
# for i in xrange(len(y_test)):
# 	stats_dict[np.argmax(predicted_output[i])] = stats_dict[np.argmax(predicted_output[i])] + 1

