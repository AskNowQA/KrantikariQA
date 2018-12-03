'''
	This file adds rdf-type constraints over the id_big_data file.
'''
from __future__ import print_function
import os
import numpy as np
import pickle, json
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils

"""
	SPARQL Templates to be used while making RDF type queries from corechains
"""
sparql_template_1 = {
	"-" : 'SELECT DISTINCT ?uri WHERE {?uri <%(r1)s> <%(te1)s> . }',
	"+" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri . }'
}

sparql_template_2 = {
	"++" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?x . ?x <%(r2)s> ?uri  . }',
	"-+" : 'SELECT DISTINCT ?uri WHERE { ?x <%(r1)s> <%(te1)s> . ?x <%(r2)s> ?uri  . }',
	"--" : 'SELECT DISTINCT ?uri WHERE { ?x <%(r1)s> <%(te1)s> . ?uri <%(r2)s> ?x  . }',
	"+-" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?x . ?uri <%(r2)s> ?x  . }'
}

sparql_template_3 = {
	"+-" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri . <te2> <%(r2)s> ?uri  . }',
	"--" : 'SELECT DISTINCT ?uri WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <%(te2)s>  . }',
	"-+" : 'SELECT DISTINCT ?uri WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <%(te2)s>  . }',
	"++" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri . ?uri <%(r2)s> <%(te2)s>  . }'
}

sparql_template_ask = {
	"+" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> <%(te2)s> . }',
    "-" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> <%(te2)s> . }'
}

x_const = '  ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_x . } '
uri_const = ' ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_uri . } '
uri_x_const = '?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_x . ' \
			  '?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_uri . } '

"""
	Macros
"""
DIR = './resources_v8/'
LCQUAD_FILE_NAME = 'id_big_data.json'
QALD_FILE_NAME_TEST = 'qald_id_big_data_test.json'
QALD_FILE_NAME_TRAIN = 'qald_id_big_data_train.json'


COMMON_DIR = 'data/data/common/'
LCQUAD_DIR = 'data/data/lcquad/'
QALD_DIR = 'data/data/qald/'

dbp = db_interface.DBPedia(_verbose=True, caching=False)

rdf_type_lookup = []



if os.path.isfile(COMMON_DIR+'rdf_type_lookup.json'):
	rdf_type_lookup = json.load(open(COMMON_DIR+'rdf_type_lookup.json'))
else:
	rdf_type_lookup = []

if os.path.isfile(COMMON_DIR+'relations.pickle'):
	relations_dict = pickle.load(open(COMMON_DIR+'relations.pickle', 'rb'), encoding='latin1')
else:
	print("error. The relation file should have existed")
	input("check and continue")
	relations_dict = {}



def random_choice(path,size):
	path_length = len(path)
	a = -1
	temp = [a for a in xrange(0,len(path))]
	temp =  np.random.choice(temp,size)
	return [path[i] for i in temp]

def retrive_relation(surface_form):
	for key in relations_dict:
		if relations_dict[key][3] == surface_form:
			relation_1 = key
			return relation_1
	return None

def check_for_constraints(_datum):
	'''


	:param _datum:  parsed data
	:return: boolean
	'''
	if '?uri' in _datum['constraints'] or '?x' in _datum['constraints']:
		return True
	else:
		return False

def relations_lookup(core_chain, core_chain_length):
	'''

	:param core_chain: corechain
	:return: converts core chain surface form to relations
		['+', 'wife','of] -> ['+','http..../wifeOf']
	'''
	if core_chain_length == 1:
		surface_form = core_chain[1:]
		relation = retrive_relation(surface_form)
		if relation:
			return [core_chain[0], relation]
		else:
			return None
	else:
		if '+' in core_chain[1:]:
			index_of = core_chain[1:].index('+')
		else:
			index_of = core_chain[1:].find('-')
		surface_form_1 = core_chain[1:index_of + 1]
		surface_form_2 = core_chain[index_of + 2 : ]
		sign_relation_2 = core_chain[index_of + 1]
		relation_1 = retrive_relation(surface_form_1)
		relation_2 = retrive_relation(surface_form_2)
		if relation_1 and relation_2:
			return [core_chain[0],relation_1,core_chain[sign_relation_2],relation_2]
		else:
			return None

def reconstruct(ent, core_chain, alternative = False):
	'''
		:param ent: an array of entity ['http://dbpedia.org/resource/Gestapo','http://dbpedia.org/resource/Gestapo']
		:param core_chain: core chain ['+','wife','of','-','president']
		:return: sparql query.
	'''



	if alternative:
		if len(core_chain) == 4:
			rel_length = 2
		else:
			rel_length = 1
	else:
		if core_chain[2].count('+') + core_chain[2].count('-') == 1:
			rel_length = 1
		else:
			rel_length = 2

	if alternative:
		relations = core_chain
	else:
		relations = relations_lookup(core_chain,rel_length)
	if not relations:
		return None
	if len(ent) == 1:
		# ent = ent[0]
		if rel_length == 1:
			sparql = sparql_template_1[core_chain[0]]
			return sparql % {'r1': str(relations[1]), 'te1':ent[0]}
		if rel_length == 2:
			if '+' in core_chain[1:]:
				index_of = core_chain[1:].index('+')
			else:
				index_of = core_chain[1:].index('-')
			sign_relation_2 = core_chain[index_of + 1]
			sparql_key = core_chain[0]+sign_relation_2
			sparql = sparql_template_2[str(sparql_key)]
			return sparql % {'r1' : str(relations[1]), 'r2' : str(relations[3]), 'te1':ent[0]}
	else:
		if rel_length == 1:
			sparql = sparql_template_ask[core_chain[0]]
			return sparql % {'r1': str(relations[1]), 'te1': ent[0], 'te2' : ent[1]}
		if '+' in core_chain[1:]:
			index_of = core_chain[1:].index('+')
		else:
			index_of = core_chain[1:].index('-')
		sign_relation_2 = core_chain[index_of + 1]
		sparql_key = core_chain[0] + sign_relation_2
		sparql = sparql_template_3[str(sparql_key)]
		return sparql % {'r1': str(relations[1]), 'r2': str(relations[3]), 'te1': ent[0],'te2':ent[1]}

def create_sparql_constraints(sparql):
	if 'SELECT DISTINCT COUNT(?uri)' not in sparql:
		sparql_x = sparql.replace('}',x_const).replace('SELECT DISTINCT ?uri', 'SELECT DISTINCT ?cons_x')
		sparql_uri = sparql.replace('}',uri_const).replace('SELECT DISTINCT ?uri', 'SELECT DISTINCT ?cons_uri')
	else:
		sparql_x = sparql.replace('}', x_const).replace('SELECT DISTINCT COUNT(?uri)', 'SELECT DISTINCT ?cons_x')
		sparql_uri = sparql.replace('}', uri_const).replace('SELECT DISTINCT COUNT(?uri)', 'SELECT DISTINCT ?cons_uri')
	return [sparql_x, sparql_uri]

def retrive_answers(sparql):
	if len(sparql) == 2:
		temp_type_x = dbp.get_answer(sparql[0])['cons_x']
		temp_type_uri = dbp.get_answer(sparql[1])['cons_uri']
		type_x = [x for x in temp_type_x if
				  x[:28] in ['http://dbpedia.org/ontology/', 'http://dbpedia.org/property/']]
		type_uri = [x for x in temp_type_uri if
					x[:28] in ['http://dbpedia.org/ontology/', 'http://dbpedia.org/property/']]
		return type_x,type_uri
	else:
		temp_type_uri = dbp.get_answer(sparql[0])['cons_uri']
		type_x = []
		type_uri = [x for x in temp_type_uri if
					x[:28] in ['http://dbpedia.org/ontology/', 'http://dbpedia.org/property/']]
		return type_x, type_uri

def generate_candidates(_datum):
	"""


	:param _datum:
	:return:
	"""

	global rdf_type_lookup

	# Check if there is some constraint or not
	data = _datum['parsed-data']
	if not check_for_constraints(data):
		return [], []

	# Get plausible candidates from DBpedia
	path = data['path']
	path = [[p[0], p[1:]] for p in path]
	path = [x for p in path for x in p]
	sparql = reconstruct(data['entity'], path, alternative=True)
	sparqls = create_sparql_constraints(sparql)
	if len(data['entity']) == 2:
		sparqls = [sparqls[1]]
	if len(path) == 2:
		sparqls = [sparqls[1]]
	type_x, type_uri = retrive_answers(sparqls)
	rdf_type_lookup = rdf_type_lookup + type_x + type_uri
	rdf_type_lookup = list(set(rdf_type_lookup))


	# Remove the "actual" constraint from this list (so that we only create negative samples)
	'''
		This does not work
	'''
	try:
		type_x = [ x for x in type_x if x not in data['constraints']['x']]
	except KeyError:
		pass

	try:
		type_uri = [ x for x in type_uri if x not in data['constraints']['uri']]
	except KeyError:
		pass

	return type_x, type_uri

def create_valid_paths(_x, _uri):
	"""
		Given the candidates of rdf type, this fn creates formatted paths.

		Grammar:
			"x/uri"
			+
			dbo:Class

		Also, converts everything to Glove Vocabulary.

	:param _x: all the possible constraints on x (intermediate variables)
	:param _uri: all the possible constraints on uri (intended answer)
	:return: two lists
	"""

	# First adopt the data into intended paths (grammar)
	type_x_candidates = []
	type_uri_candidates = []
	for x in _x:

		# Model this thing into a path
		path = "x + " + dbp.get_label(x)
		path_id = embeddings_interface.vocabularize(nlutils.tokenize(path))
		type_x_candidates.append(path_id)
	for uri in _uri:

		# Model this thing into a path
		path = "uri + " + dbp.get_label(uri)
		path_id = embeddings_interface.vocabularize(nlutils.tokenize(path))
		type_uri_candidates.append(path_id)

	return type_x_candidates,type_uri_candidates

def run():
	"""
		Main script which calls and does everything.
	:return: nothing
	"""
	raw_data = json.load(open(LCQUAD_DIR+LCQUAD_FILE_NAME,'r'))

	for i in range(0,len(raw_data)):
		x,uri = generate_candidates(raw_data[i])
		type_x_candidates, type_uri_candidates = create_valid_paths(x,uri)
		raw_data[i]['rdf-type-constraints'] = type_x_candidates + type_uri_candidates
		print(i)

	for i in range(0,len(raw_data)):
		for j in range(0,len(raw_data[i]['rdf-type-constraints'])):
			raw_data[i]['rdf-type-constraints'][j] = raw_data[i]['rdf-type-constraints'][j].tolist()

	json.dump(raw_data,open(LCQUAD_DIR+LCQUAD_FILE_NAME,'w+'))
	json.dump(rdf_type_lookup, open(COMMON_DIR+'rdf_type_lookup.json','w+'))


def qald_run(test = True):
	''''

		If the query contains only rdf constraint and no other triple, it will push the rdf type to the
		constraint candidate  and there will be no positive or negative path in the query.

		'path_id': [u'+25212']
	'''

	if test:
		raw_data = json.load(open(QALD_DIR+QALD_FILE_NAME_TEST,'r'))
	else:
		raw_data = json.load(open(QALD_DIR+QALD_FILE_NAME_TRAIN,'r'))


	'''
		Find the rdf-type id
	'''

	rdf_type_id = None
	for data in raw_data:
		if data['parsed-data']['path'][0] == -1:
			continue
		if data['parsed-data']['path'][0][1:] == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
			rdf_type_id = data['parsed-data']['path_id'][1:]

	for i in range(0,len(raw_data)):
		if raw_data[i]['parsed-data']['path_id'][0] == -1:
			raw_data[i]['rdf-type-constraints'] = []
		elif raw_data[i]['parsed-data']['path_id'][0][1:] == rdf_type_id:
			raw_data[i]['rdf-type-constraints'] = []
		else:
			x, uri = generate_candidates(raw_data[i])
			type_x_candidates, type_uri_candidates = create_valid_paths(x, uri)
			raw_data[i]['rdf-type-constraints'] = type_x_candidates + type_uri_candidates
			print(i)

	for i in range(0,len(raw_data)):
		for j in range(0,len(raw_data[i]['rdf-type-constraints'])):
			raw_data[i]['rdf-type-constraints'][j] = raw_data[i]['rdf-type-constraints'][j].tolist()

	if test:
		json.dump(raw_data,open(QALD_DIR+QALD_FILE_NAME_TEST,'w+'))
		json.dump(rdf_type_lookup, open(COMMON_DIR + 'rdf_type_lookup.json', 'w+'))
	else:
		json.dump(raw_data, open(QALD_DIR+QALD_FILE_NAME_TRAIN, 'w+'))
		json.dump(rdf_type_lookup, open(COMMON_DIR + 'rdf_type_lookup.json', 'w+'))

if __name__ == "__main__":
	qald_run(test=True)

