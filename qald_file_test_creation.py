'''

	The file create a test set for qald.

'''



import sys
import json
import pickle
import warnings
import traceback
import numpy as np
import editdistance
from pprint import pprint
from progressbar import ProgressBar

# Local file imports
import krantikari as K
import krantikari_new as KN
import qald_parser as qp
from utils import model_interpreter
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils


dbp = db_interface.DBPedia(_verbose=True, caching=False)

TEST = True
DEBUG = True
exception_log = []
big_data_test = []

log = []


INIT_LCQUAD_FILE_NAME = 'big_data.pickle'
INIT_QALD_FILE_NAME_TEST = 'qald_big_data_test.pickle'
INIT_QALD_FILE_NAME_TRAIN = 'qald_big_data_training.pickle'


COMMON_DIR = 'data/data/common/'
LCQUAD_DIR = 'data/data/lcquad/'
QALD_DIR = 'data/data/qald/'



'''
	>Load data.
'''
if TEST:
	raw_dataset = json.load(open(qp.RAW_QALD_DIR_TEST))['questions']
	parsed_dataset = pickle.load(open(qp.PARSED_QALD_DIR_TEST))
else:
	raw_dataset = json.load(open(qp.RAW_QALD_DIR_TRAIN))['questions']
	parsed_dataset = pickle.load(open(qp.PARSED_QALD_DIR_TRAIN))


# Iterate through every question
c = 0
for i in range(len(raw_dataset[c:])):
	try:
		logger = {}
		logger['id'] = i
		logger['raw_dataset'] = raw_dataset[i]

		#Stores all the data and meta data for a particular node
		temp_big_data = {}
		# Get the QALD question
		q_raw = raw_dataset[i]
		q_parsed = parsed_dataset[i]
		logger['q_raw'] = q_raw
		logger['q_parsed'] = q_parsed


		if DEBUG:
			print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
			print i+c

		# # Get answer for the query
		# ans = dbp.get_answer(q_raw['query']['sparql'])

		# true_path, topic_entities = get_true_path(q_parsed, q_raw['query']['sparql'])
		data = qp.get_true_path(q_parsed, q_raw['query']['sparql'])
		logger['data'] = data


		if data[1] == -1 or data[0] is None:
			log.append(logger)
			continue



		question = q_raw['question'][0]['string']
		entity = data[1]

		logger['question'] = question
		logger['entity'] = entity

		is_uri = True
		for e in entity:
			if not nlutils.is_dbpedia_uri(e):
				logger['is_uri'] = True
				is_uri = False
		if not is_uri:
			logger.append['is_uri'] = False
			log.append(logger)
			continue
		if "ask" in q_raw['query']['sparql'].lower():
			logger['is_ask'] = True
			training_data = KN.Krantikari_v2(_question=question, _entities=entity, _model_interpreter="", _dbpedia_interface=dbp,
										  _training=True, _ask=True,_qald=True)
		else:
			logger['is_ask'] = False
			training_data = KN.Krantikari_v2(_question=question, _entities=entity, _model_interpreter="", _dbpedia_interface=dbp,
										  _training=True, _ask=False,_qald=True)

		temp_big_data['unparsed-qald-data'] = q_raw
		temp_big_data['parsed-qald-data'] = q_parsed
		temp_big_data['uri'] = training_data.data
		logger['big_data_node'] = temp_big_data
		logger['exception'] = ''
		log.append(logger)
		big_data_test.append(temp_big_data)
	except:
		print traceback.print_exc()
		logger = {}
		logger['id'] = i
		logger['raw_dataset'] = raw_dataset[i]
		logger['exception'] = traceback.print_exc()
		log.append(logger)
		exception_log.append(raw_dataset[i])


if TEST:
	pickle.dump(big_data_test,open(QALD_DIR + INIT_QALD_FILE_NAME_TEST,'w+'))
	pickle.dump(log,open(QALD_DIR+'log_test.pickle','w+'))
else:
	pickle.dump(big_data_test,open(QALD_DIR + INIT_QALD_FILE_NAME_TRAIN,'w+'))
	pickle.dump(log,open(QALD_DIR+'log_train.pickle','w+'))