'''
	The script tries to parse the 30MQA file and generate 10000 factoid samples.
'''
import random
import os
import json
from pprint import pprint
# Custom files
import utils.dbpedia_interface as db_interface


FALSE_PATH_LENGTH = 20
REQUIRED_QUESTION = 22
WRITE_INTERVAL = 10
OUTPUT_DIR = 'data/30MQA'
SPARQL = '''SELECT DISTINCT ?entity ?label WHERE { ?entity owl:sameAs %(target_resource)s . ?entity <http://www.w3.org/2000/01/rdf-schema#label> ?label . FILTER (lang(?label) = 'en') } '''


fname = 'resources/30MQA_1/fqFiltered.txt'
dbp = db_interface.DBPedia(_verbose=True, caching=False)


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
		final_data.append([(value['entity'],value['label']),con[1],con[3],false_path])
		periodic_counter = periodic_counter + 1
		print "write counter is ", counter
		counter = counter + 1
		if periodic_counter > WRITE_INTERVAL:
			with open(OUTPUT_DIR + "/" + str(content_counter) + ".json", 'w') as fp:
				json.dump(final_data, fp)
			periodic_counter = 0
			final_data = []
		content_counter = content_counter + 1
		if counter > REQUIRED_QUESTION:
			break
	else:
		content_counter = content_counter + 1
		print content_counter
if len(final_data) > 0:
	with open(OUTPUT_DIR + "/" + 'remaining.json', 'w') as fp:
		json.dump(final_data, fp)


