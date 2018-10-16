'''
	The script tries to parse the 30MQA file and generate 10000 factoid samples.
'''
import os
import json
import pickle
import random
import numpy as np
# Custom files
import utils.dbpedia_interface as db_interface
import utils.natural_language_utilities as nlutils
from utils import embeddings_interface

embeddings_interface.__prepare__(False, True)


FALSE_PATH_LENGTH = 50
REQUIRED_QUESTION = 10000
WRITE_INTERVAL = REQUIRED_QUESTION	#No interm write
OUTPUT_DIR = 'data/30MQA'
SPARQL = '''SELECT DISTINCT ?entity ?label WHERE { ?entity owl:sameAs %(target_resource)s . ?entity <http://www.w3.org/2000/01/rdf-schema#label> ?label . FILTER (lang(?label) = 'en') } '''


fname = 'resources/30MQA_1/fqFiltered.txt'
dbp = db_interface.DBPedia(_verbose=True, caching=True)


try:
    os.makedirs(OUTPUT_DIR)
except OSError:
    print("Folder already exists")



def random_indice_generator(_length,_false_path_length,content_counter):

	indices = []
	for i in xrange(_false_path_length):
		value = random.randint(0,_length)
		if value == content_counter:
			value = random.randint(0,_length)
		indices.append(value)
	return indices


def vectorize_relation(path):
	path = path.split('/')
	path = path[1:]
	path = [" ".join(i.split('_')) for i in path]
	path = [nlutils.tokenize(x) for x in path]
	path = [x for y in path for x in y]
	path = embeddings_interface.vectorize(path)
	return path

with open(fname) as f:
	content = f.readlines()

content = content[0:100000]
content = [con.replace('\n','').split('\t') for con in content]


final_data = []
processed_question = 0
content_counter = 0
periodic_counter = 0
counter = 0
for con in content:
	try:
		value = dbp.get_answer(SPARQL %{'target_resource' : con[0]})
		if value['label'] and value['entity']:
			indices = random_indice_generator(len(content),FALSE_PATH_LENGTH,content_counter)
			for i in xrange(len(indices)):
				condition = True
				while condition:
					try:
						a = content[i][1]
					except:
						print content
					if content[indices[i]][1] == con[1]:
						indices[i] = random.randint(0,len(content))
					if content[indices[i]][1] != con[1]:
						condition = False
			false_path = [content[i][1] for i in indices]
			tokenized_false_path = [vectorize_relation(path) for path in false_path]
			question_vector = embeddings_interface.vectorize(nlutils.tokenize(con[3]))
			entity_vector = embeddings_interface.vectorize(nlutils.tokenize(value['label'][0]))
			relation_vector = vectorize_relation(con[1])
			plus_vector = embeddings_interface.vectorize(['+'])
			entity_vector = np.concatenate((entity_vector,plus_vector),axis=0)
			new_entity_vector = np.concatenate((entity_vector,relation_vector),axis=0)
			y_vector = np.zeros(FALSE_PATH_LENGTH + 1)
			final_data.append([question_vector,new_entity_vector,tokenized_false_path,y_vector])
			periodic_counter = periodic_counter + 1
			print "write counter is ", counter
			counter = counter + 1
			if periodic_counter > WRITE_INTERVAL:
				with open(OUTPUT_DIR + "/" + str(content_counter) + ".pickle", 'w') as fp:
					pickle.dump(final_data, fp)
				periodic_counter = 0
				final_data = []
			content_counter = content_counter + 1
			if counter > REQUIRED_QUESTION:
				break
		else:
			content_counter = content_counter + 1
			print content_counter
	except:
		content_counter = content_counter + 1
		continue


if len(final_data) > 0:
	with open(OUTPUT_DIR + "/" + 'remaining.json', 'w') as fp:
		json.dump(final_data, fp)


