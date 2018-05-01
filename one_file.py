'''
	This script will run the whole pipeline at testing time. Following model would be used
		> Core chain ranking model - ./data/training/pairwise/model_30/model.h5
		> Classifier for ask and count. - still need to be trained
		> rdf type classifier - still need to be trained
'''

from __future__ import absolute_import

import os
from utils import embeddings_interface
from utils import natural_language_utilities as nlutils

gpu = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import sys
import json
import math
import pickle
import numpy as np
import pandas as pd
from keras import optimizers
import keras.backend.tensorflow_backend as K
from keras.preprocessing.sequence import pad_sequences


# Some Macros
DEBUG = True
DATA_DIR = './data/training/pairwise'
EPOCHS = 300
BATCH_SIZE = 880 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
NEGATIVE_SAMPLES = 1000
OPTIMIZER = optimizers.Adam(LEARNING_RATE)

relations = pickle.load(open('resources_v8/relations.pickle'))
reverse_relations = {}
for keys in relations:
	reverse_relations[relations[keys][0]] = [keys] + relations[keys][1:]


def custom_loss(y_true, y_pred):
	'''
		max margin loss
	'''
	# y_pos = y_pred[0]
	# y_neg= y_pred[1]
	diff = y_pred[:,-1]
	# return K.sum(K.maximum(1.0 - diff, 0.))
	return K.sum(diff)

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

def rank_precision_runtime(model, id_q, id_tp, id_fps, batch_size=1000, max_length=50):
	'''
		A function to pad the data for the model, run model.predict on it and get the resuts.

	:param id_q: A 1D array of the question
	:param id_tp: A 1D array of the true path
	:param id_fps: A list of 1D arrays of false paths
	:param batch_size: int: the batch size the model expects
	:param max_length: int: size with which we pad the data
	:return: ?? @TODO
	'''

	# Create empty matrices
	question = np.zeros((len(id_fps)+1, max_length))
	paths = np.zeros((len(id_fps)+1, max_length))

	# Fill them in
	question[:, id_q.shape[0]] = np.repeat(id_q[np.newaxis,:], question.shape[0], axis=0)
	paths[0, :id_tp.shape[0]] = id_tp
	for i in range(len(id_fps)):
		paths[i+1,:id_fps[i].shape[0]] = id_fps[i]

	# Pass em through the model
	results = model.predict([question, paths, np.zeros_like(paths)], batch_size=batch_size)[:,0]
	return results

def rank_precision_metric(neg_paths_per_epoch):
	def metric(y_true, y_pred):
		scores = y_pred[:, 0]
		scores = K.reshape(scores, (-1, neg_paths_per_epoch+1))
		hits = K.cast(K.shape(K.tf.where(K.tf.equal(K.tf.argmax(scores, axis=1),0)))[0], 'float32')
		precision = hits/K.cast(K.shape(scores)[0], 'float32')
		# precision = float(len(np.where(np.argmax(all_outputs, axis=1)==0)[0]))/all_outputs.shape[0]
		return precision
	return metric

def get_glove_embeddings():
	from utils.embeddings_interface import __check_prepared__
	__check_prepared__('glove')

	from utils.embeddings_interface import glove_embeddings
	return glove_embeddings

def cross_correlation(x):
	a, b = x
	tf = K.tf
	a_fft = tf.fft(tf.complex(a, 0.0))
	b_fft = tf.fft(tf.complex(b, 0.0))
	ifft = tf.ifft(tf.conj(a_fft) * b_fft)
	return tf.cast(tf.real(ifft), 'float32')

def load_data(file, max_sequence_length):
	glove_embeddings = get_glove_embeddings()

	try:
		with open(os.path.join(DATA_DIR, file + ".mapped.npz")) as data, open(os.path.join(DATA_DIR, file + ".index.npy")) as idx:
			dataset = np.load(data)
			questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
			index = np.load(idx)
			# vectors = glove_embeddings[index]
			return None, questions, pos_paths, neg_paths
	except:
		with open(os.path.join(DATA_DIR, file)) as fp:
			dataset = pickle.load(fp)
			questions = [i[0] for i in dataset]
			questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
			pos_paths = [i[1] for i in dataset]
			pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
			neg_paths = [i[2] for i in dataset]
			neg_paths = [path for paths in neg_paths for path in paths]
			neg_paths = pad_sequences(neg_paths, maxlen=max_sequence_length, padding='post')

			all = np.concatenate([questions, pos_paths, neg_paths], axis=0)
			mapped_all, index = pd.factorize(all.flatten(), sort=True)
			mapped_all = mapped_all.reshape((-1, max_sequence_length))
			vectors = glove_embeddings[index]

			questions, pos_paths, neg_paths = np.split(mapped_all, [questions.shape[0], questions.shape[0]*2])
			neg_paths = np.reshape(neg_paths, (len(questions), NEGATIVE_SAMPLES, max_sequence_length))

			with open(os.path.join(DATA_DIR, file + ".mapped.npz"), "w") as data, open(os.path.join(DATA_DIR, file + ".index.npy"), "w") as idx:
				np.savez(data, questions, pos_paths, neg_paths)
				np.save(idx, index)

			return vectors, questions, pos_paths, neg_paths



def create_data_big_data(data,keras=True):
	'''
		The function takes id version of big data node and transforms it to version required by Keras network code.
	'''

	false_paths = []
	true_path = []
	id_paths = []
	for rel in data['parsed-data']['path_id']:
		true_path = true_path + [rel[0]] + reverse_relations[int(rel[1:])][-2]
		id_paths.append(data['parsed-data']['path_id'])
	for rel in data['uri']['hop-1-properties']:
		temp_path = [str(rel[0])] +  reverse_relations[int(rel[1])][-2]
		false_paths.append(temp_path)
		id_paths.append(temp_path)
	for rel in data['uri']['hop-2-properties']:
		temp_path = [str(rel[0])] +  reverse_relations[int(rel[1])][-2] + [str(rel[2])] +  reverse_relations[int(rel[3])][-2]
		false_paths.append(temp_path)
		id_paths.append(temp_path)
	question = data['parsed-data']['corrected_question']
	return [question,true_path,false_paths,id_paths]

max_length = 50

# Shuffle these matrices together @TODO this!
np.random.seed(0) # Random train/test splits stay the same between runs

# Divide the data into diff blocks
split_point = lambda x: int(len(x) * .80)

def train_split(x):
	return x[:split_point(x)]
def test_split(x):
	return x[split_point(x):]

with K.tf.device('/gpu:' + gpu):
	from keras.models import load_model
	metric = rank_precision_metric(10)
	model_corechain = load_model("./data/training/pairwise/model_30/model.h5", {'custom_loss':custom_loss, 'metric':metric})
	model_rdf_type_check = load_model("./data/training/rdf/model_00/model.h5")



# counter = 0
# for i in xrange(0,len(test_questions)):
#     counter = counter + rank_precision(model, test_questions[i].reshape(1,-1), test_pos_paths[i].reshape(1,-1), test_neg_paths[i].reshape(1,1000,-1), 1000, 10000)

def rdf_type_check(question,model_rdf_type_check, max_length = 30):
	'''

	:param question: vectorize question
	:param model_rdf_type_check: model
	:return:
	'''
	question_padded = np.zeros((1,max_length))
	try:
		question_padded[:,:question.shape[0]] = question
	except ValueError:
		question_padded = question[:,:question_padded.shape[0]]
	prediction = np.argmax(model_rdf_type_check.predict(question_padded))
	if prediction == 0:
		return True
	else:
		return False



id_data = json.load(open('resources_v8/id_big_data.json'))
id_data_test = test_split(id_data)

core_chain_correct = []

for data in id_data_test:
	max_sequence_length = 50
	result = create_data_big_data(data)
	id_q = embeddings_interface.vocabularize(nlutils.tokenize(result[0]), _embedding="glove")
	id_tp = embeddings_interface.vocabularize(result[1])
	id_fps = [embeddings_interface.vocabularize(x) for x in result[2]]
	output = rank_precision_runtime(model_corechain, id_q, id_tp,
				   id_fps, 10000, 50)
	if np.argmax(output[:,0]) == 0:
		core_chain_correct.append(data)
	max_index = np.argmax(output[:,0])
	id_path = result[3][max_index]
	if rdf_type_check(id_q,model_rdf_type_check):
		#generate rdf candidates
		print "done"
	else:
		#no rdf type candidate
		print "not done"
