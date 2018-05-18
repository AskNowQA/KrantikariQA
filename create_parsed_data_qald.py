'''
	This file will take the big_data file made by the krantikari and then create the 'parsed-data' field similar to LC-QuAD.
	If there does not exists any positive path then it add a -1 at 'path'
'''


import json
import pickle


import qald_parser as qp

DEBUG = False


TEST = True


FINAL_QALD_FILE_NAME_TEST = 'qald_big_data_test.json'
FINAL_QALD_FILE_NAME_TRAIN = 'qald_big_data_train.json'


INIT_LCQUAD_FILE_NAME = 'big_data.pickle'
INIT_QALD_FILE_NAME_TEST = 'qald_big_data_test.pickle'
INIT_QALD_FILE_NAME_TRAIN = 'qald_big_data_training.pickle'


COMMON_DIR = 'data/data/common/'
LCQUAD_DIR = 'data/data/lcquad/'
QALD_DIR = 'data/data/qald/'




if TEST:
	big_data_test = pickle.load(open(QALD_DIR+INIT_QALD_FILE_NAME_TEST))
else:
	big_data_train = pickle.load(open(QALD_DIR+INIT_QALD_FILE_NAME_TRAIN))

if DEBUG:
	big_data_test = pickle.load(open('data/data/qald/qald_big_data_test_v2.pickle'))

def parsed_data(node):
	'''
	:param node: big data node
	:return: parsed-data field

	'''
	parsed_data = {}
	data = qp.get_true_path(node['parsed-qald-data'], node['unparsed-qald-data']['query']['sparql'])
	parsed_data['constraints'] = data[2]
	parsed_data['corrected_question'] = node['unparsed-qald-data']['question'][0]['string']
	parsed_data['entity'] = data[1]
	parsed_data['path'] = data[0]
	'''
		data specific hack
	'''
	try:
		if data[0][0][1:] == 'http://dbpedia.org/ontology/type':
			parsed_data['path'] = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
	except:
		parsed_data['path'] = [-1]
	parsed_data['sparql_query'] = node['unparsed-qald-data']['query']['sparql']
	return parsed_data

if not DEBUG:
	if not TEST:
		for i in range(0,len(big_data_train)):
			big_data_train[i]['parsed-data'] = parsed_data(big_data_train[i])
	else:
		for i in range(0,len(big_data_test)):
			big_data_test[i]['parsed-data'] = parsed_data(big_data_test[i])

if not DEBUG:
	if TEST:
		pickle.dump(big_data_test,open('data/data/qald/qald_big_data_test.pickle','w+'))
	else:
		pickle.dump(big_data_train,open('data/data/qald/qald_big_data_training.pickle','w+'))
else:
	pickle.dump(big_data_test,open('data/data/qald/qald_big_data_test_v2.pickle','w+'))


if not DEBUG:
	if not TEST:
		for i in range(0,len(big_data_train)):
			big_data_train[i]['uri']['question-id'] = list(big_data_train[i]['uri']['question-id'])
	else:
		for i in range(0,len(big_data_test)):
			big_data_test[i]['uri']['question-id'] = list(big_data_test[i]['uri']['question-id'])

if not DEBUG:
	if TEST:
		json.dump(big_data_test,open(QALD_DIR+FINAL_QALD_FILE_NAME_TEST,'w+'))
	else:
		json.dump(big_data_train,open(QALD_DIR+FINAL_QALD_FILE_NAME_TRAIN,'w+'))
else:
	json.dump(big_data_test,open('data/data/qald/qald_big_data_test_v2.json','w+'))