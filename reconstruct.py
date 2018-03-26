'''
	This will re-construct SPARQL given core-chain and topic entites.
'''

import pickle, json
from utils import dbpedia_interface as db_interface
import krantikari as K

BIG_DATA_DIR = 'resources/big_data.pickle'

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
	"++" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri . <te2> <%(r2)s> ?uri  . }',
	"--" : 'SELECT DISTINCT ?uri WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <te2>  . }',
	"-+" : 'SELECT DISTINCT ?uri WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <te2>  . }'
}

x_const = '  ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_x . } '
uri_const = ' ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_uri . } '
uri_x_const = '?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_x . ' \
			  '?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_uri . } '


relations_dict = pickle.load(open('resources/relations.pickle'))
dbp = db_interface.DBPedia(_verbose=True, caching=True)

def retrive_relation(surface_form):
	for key in relations_dict:
		if relations_dict[key][3] == surface_form:
			relation_1 = key
			return relation_1
	return None

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
			return sparql % {'r1': str(relations[1:]), 'te1':ent[0]}
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
		if '+' in core_chain[1:]:
			index_of = core_chain[1:].index('+')
		else:
			index_of = core_chain[1:].index('-')
		sign_relation_2 = core_chain[index_of + 1]
		sparql_key = core_chain[0] + sign_relation_2
		sparql = sparql_template_3[str(sparql_key)]
		return sparql % {'r1': str(relations[1]), 'r2': str(relations[3]), 'te1': ent[0],'te2':ent[1]}



def create_sparql_constraints(sparql):
	sparql_x = sparql.replace('}',x_const).replace('SELECT DISTINCT ?uri', 'SELECT DISTINCT ?cons_x')
	sparql_uri = sparql.replace('}',uri_const).replace('SELECT DISTINCT ?uri', 'SELECT DISTINCT ?cons_uri')
	# sparql_uri_x = sparql.replace('}',uri_x_const).replace('SELECT DISTINCT ?uri', 'SELECT DISTINCT ?cons_uri, ?cons_x')
	return [sparql_x, sparql_uri]


def retrive_answers(sparql):
	temp_type_x = dbp.get_answer(sparql[0])['cons_x']
	temp_type_uri = dbp.get_answer(sparql[1])['cons_uri']
	type_x = [x for x in temp_type_x if
			  x[:28] in ['http://dbpedia.org/ontology/', 'http://dbpedia.org/property/']]
	type_uri = [x for x in temp_type_uri if
				x[:28] in ['http://dbpedia.org/ontology/', 'http://dbpedia.org/property/']]
	return type_x,type_uri

def construct_training_data():
	'''
		Find the core chain, create constraints and then add it  to the training data set.
		The constraints would be
			>rdf type on x
			>rdf type on uri
			>rdf type on x and uri
	'''
	# big_data = pickle.load(open(BIG_DATA_DIR))
	# for data in big_data:
	# 	path_id = data['parsed-data']['path_id']
	# 	if len(path_id) > 1:
	# 		path_id = [path_id[0][0], path_id[0][1:], path_id[1][0], path_id[1][1:]]
	# 	else:
	# 		path_id = [path_id[0][0], path_id[0][1:]]
	parsed_data = []
	for data in parsed_data:
		path = data['path']
		path = [[p[0],p[1:]] for p in path]
		path = [x for p in path for x in p]
		sparql = reconstruct(data['parsed-data']['entity'],path, alternative=True)
		sparqls = create_sparql_constraints(sparql)
		type_x,type_uri = retrive_answers(sparqls)
		data['constraints_candidates'] = {
			"type_x" : type_x,
			"type_uri" : type_uri
		}


data_set = json.load(open('resources/data_set.json'))
new_data = []
counter = 0

for x in data_set:
	try:
		data = K.parse_lcquad(x)
		path = data['path']
		path = [[p[0], p[1:]] for p in path]
		path = [x for p in path for x in p]
		sparql = reconstruct(data['entity'], path, alternative=True)
		sparqls = create_sparql_constraints(sparql)
		type_x, type_uri = retrive_answers(sparqls)
		data['constraints_candidates'] = {
			"type_x": type_x,
			"type_uri": type_uri
		}
		new_data.append(data)
		print counter
		counter = counter + 1
	except:
		continue

pickle.dump(new_data,open('resources/rdf-type.pickle','w+'))