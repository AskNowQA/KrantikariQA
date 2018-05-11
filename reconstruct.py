'''
	This will re-construct SPARQL given core-chain and topic entites.
'''

import numpy as np
import krantikari as K
import pickle, json, traceback
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils

# BIG_DATA_DIR = 'resources/big_data.pickle'
DIR = './resources_v8/'

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
	"-+" : 'SELECT DISTINCT ?uri WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <%(te2)s>  . }'
}

sparql_template_ask = {
	"+" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> <%(te2)s> . }'
}

x_const = '  ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_x . } '
uri_const = ' ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_uri . } '
uri_x_const = '?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_x . ' \
			  '?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_uri . } '


relations_dict = pickle.load(open('resources/relations.pickle'))
dbp = db_interface.DBPedia(_verbose=True, caching=True)



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

def idfy(training_data_v2,name):
	MAX_FALSE_PATHS = 270

	id_results = []

	counter = 0
	for result in training_data_v2:
		# Id-fy the entire thing
		try:
			id_q = embeddings_interface.vocabularize(nlutils.tokenize(result[0]), _embedding="glove")
			id_tp = embeddings_interface.vocabularize(result[2])
			id_fps = [embeddings_interface.vocabularize(x) for x in result[3]]

			# Actual length of False Paths
			# actual_length_false_path.append(len(id_fps))

			# Makes the number of Negative Samples constant
			# id_fps = np.random.choice(id_fps, size=MAX_FALSE_PATHS)
			id_fps = random_choice(id_fps, size=MAX_FALSE_PATHS)

			# Make neat matrices.
			id_results.append([id_q, id_tp, id_fps, np.zeros((20, 1))])
		except:
			'''
				There is some bug in random choice. Need to investigate more on this.
			'''
			print traceback.print_exc()
			raw_input()
			counter = counter + 1
	# embeddings_interface.save_out_of_vocab()
	pickle.dump(id_results, open(DIR + name, 'w+'))

def main():
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
			if len(data['entity']) == 2 :
				sparqls = [sparqls[1]]
			if len(path) == 2:
				sparqls = [sparqls[1]]
			type_x, type_uri = retrive_answers(sparqls)
			data['constraints_candidates'] = {
				"type_x": type_x,
				"type_uri": type_uri
			}
			new_data.append(data)
			print counter
			counter = counter + 1
		except:
			print traceback.print_exc()
			# raw_input()
			continue

	pickle.dump(new_data,open('resources/rdf-type.pickle','w+'))
	rdf = pickle.load(open('resources/rdf-type.pickle'))
	new_data = []
	t = 0
	for data in rdf:
		try:
			temp = data
			# pprint(data)
			const = data['constraints']
			path = ""
			try:
				path = const['?uri']
				t= t + 1
			except:
				pass
			if not path:
				try:
					path = const['?x']
				except:
					pass
			temp['path-constraint'] = path
			new_data.append(temp)
			# raw_input()
		except:
			continue

	error = []
	counter = 0
	for data in new_data:
		if data['path-constraint'] :
			if data['path-constraint'] not in data['constraints_candidates']['type_uri'] and data['path-constraint'] not in data['constraints_candidates']['type_x']:
				counter = counter + 1
				error.append(data)

	template_dict = {}

	for x in error:
		if x['sparql_template_id'] in template_dict:
			template_dict[x['sparql_template_id']] = template_dict[x['sparql_template_id']] + 1
		else:
			template_dict[x['sparql_template_id']] = 0
	return rdf

def strategy_1(new_data,id=True):
	training_data = []
	training_data_v2 = []
	true_paths = []
	false_paths = []
	for data in new_data:
		temp = data
		true_path = ''
		false_path = []
		path = path = [[p[0], p[1:]] for p in data['path']]
		path = [x for p in path for x in p]

		if data['path-constraint'] != "":
			if '?uri' in data['constraints']:
				'''
					Remove the true path from the false path
				'''
				try:
					data['constraints_candidates']['type_uri'].remove(data['path-constraint'])
				except:
					print traceback.print_exc()
					continue
				#Still need to create this path
				true_path = path + ['/'] +  ['uri'] + [data['path-constraint']]

			else:
				try:
					data['constraints_candidates']['type_x'].remove(data['path-constraint'])

				except:
					print traceback.print_exc()
					continue
				true_path = path + ['/'] + ['x'] +[data['path-constraint']]

			for d in data['constraints_candidates']['type_uri']:
				false_path.append(path + ['/'] + ['uri'] + [d])
			for d in data['constraints_candidates']['type_x']:
				false_path.append(path + ['/'] + ['x'] + [d])
			temp['true-path'] = true_path
			temp['false-path'] = false_path
			label_true_path = []
			for token in true_path:
				if token == '/':
					label_true_path.append('/')
				else:
					label_true_path.append(nlutils.tokenize(dbp.get_label(token))[0])
			temp['label-true-path'] = label_true_path
			label_false_paths = []
			for path in false_path:
				temp_fp = []
				for token in path:
					if token == '/':
						temp_fp.append('/')
					else:
						temp_fp.append(nlutils.tokenize(dbp.get_label(token))[0])
				label_false_paths.append(temp_fp)
			temp['label-false-path'] = label_false_paths
			false_paths.append(false_path)
			true_paths.append(true_path)
			training_data.append(temp)
			training_data_v2.append([data[u'corrected_question'],data['entity'],label_true_path,label_false_paths,data['_id']])
	# pickle.dump(training_data,open('resources/training-rdf-data.pickle','w+'))
	if id:
		idfy(training_data_v2, 'training-rdf-data-v1.pickle')
	else:
		pickle.dump(training_data_v2,open(DIR+ 'training-rdf-data-v1.pickle','w+'))

def strategy_2(new_data,id=True):
	training_data = []
	training_data_v2 = []
	true_paths = []
	false_paths = []
	for data in new_data:
		temp = data
		true_path = ''
		false_path = []
		path = path = [[p[0], p[1:]] for p in data['path']]
		path = [x for p in path for x in p]

		if data['path-constraint'] != "":
			if '?uri' in data['constraints']:
				'''
					Remove the true path from the false path
				'''
				try:
					data['constraints_candidates']['type_uri'].remove(data['path-constraint'])
				except:
					print traceback.print_exc()
					continue
				#Still need to create this path
				true_path = ['uri'] + [data['path-constraint']]

			else:
				try:
					data['constraints_candidates']['type_x'].remove(data['path-constraint'])

				except:
					print traceback.print_exc()
					continue
				true_path = ['x'] +[data['path-constraint']]

			for d in data['constraints_candidates']['type_uri']:
				false_path.append(['uri'] + [d])
			for d in data['constraints_candidates']['type_x']:
				false_path.append(['x'] + [d])
			temp['true-path'] = true_path
			temp['false-path'] = false_path
			label_true_path = []
			for token in true_path:
				if token == '/':
					label_true_path.append('/')
				else:
					label_true_path.append(nlutils.tokenize(dbp.get_label(token))[0])
			temp['label-true-path'] = label_true_path
			label_false_paths = []
			for path in false_path:
				temp_fp = []
				for token in path:
					if token == '/':
						temp_fp.append('/')
					else:
						temp_fp.append(nlutils.tokenize(dbp.get_label(token))[0])
				label_false_paths.append(temp_fp)
			temp['label-false-path'] = label_false_paths
			false_paths.append(false_path)
			true_paths.append(true_path)
			training_data.append(temp)
			training_data_v2.append([data[u'corrected_question'],data['entity'],label_true_path,label_false_paths,data['_id']])

	if id:
		idfy(training_data_v2, 'training-rdf-data-v2.pickle')
	else:
		pickle.dump(training_data_v2,open(DIR+'training-rdf-data-v2.pickle','w+'))

def strategy_3(new_data,id=True):
	training_data = []
	training_data_v2 = []
	true_paths = []
	false_paths = []
	for data in new_data:
		temp = data
		true_path = ''
		false_path = []
		path = path = [[p[0], p[1:]] for p in data['path']]
		path = [x for p in path for x in p]

		if data['path-constraint'] != "":
			if '?uri' in data['constraints']:
				'''
					Remove the true path from the false path
				'''
				try:
					data['constraints_candidates']['type_uri'].remove(data['path-constraint'])
				except:
					print traceback.print_exc()
					continue
				#Still need to create this path
				true_path = [data['path-constraint']]

			else:
				try:
					data['constraints_candidates']['type_x'].remove(data['path-constraint'])

				except:
					print traceback.print_exc()
					continue
				true_path = [data['path-constraint']]

			for d in data['constraints_candidates']['type_uri']:
				false_path.append([d])
			for d in data['constraints_candidates']['type_x']:
				false_path.append([d])
			temp['true-path'] = true_path
			temp['false-path'] = false_path
			label_true_path = []
			for token in true_path:
				if token == '/':
					label_true_path.append('/')
				else:
					label_true_path.append(nlutils.tokenize(dbp.get_label(token))[0])
			temp['label-true-path'] = label_true_path
			label_false_paths = []
			for path in false_path:
				temp_fp = []
				for token in path:
					if token == '/':
						temp_fp.append('/')
					else:
						temp_fp.append(nlutils.tokenize(dbp.get_label(token))[0])
				label_false_paths.append(temp_fp)
			temp['label-false-path'] = label_false_paths
			false_paths.append(false_path)
			true_paths.append(true_path)
			training_data.append(temp)
			training_data_v2.append([data[u'corrected_question'],data['entity'],label_true_path,label_false_paths,data['_id']])
	if id:
		idfy(training_data_v2, 'training-rdf-data-v3.pickle')
	else:
		pickle.dump(training_data_v2,open(DIR+'training-rdf-data-v3.pickle','w+'))

def point_wise_strategy_1(data):
		temp = data
		true_path = ''
		false_path = []
		path = path = [[p[0], p[1:]] for p in data['path']]
		path = [x for p in path for x in p]

		if data['path-constraint'] != "":
			if '?uri' in data['constraints']:
				'''
					Remove the true path from the false path
				'''
				try:
					data['constraints_candidates']['type_uri'].remove(data['path-constraint'])
				except:
					print traceback.print_exc()
				#Still need to create this path
				true_path = path + ['/'] +  ['uri'] + [data['path-constraint']]

			else:
				try:
					data['constraints_candidates']['type_x'].remove(data['path-constraint'])

				except:
					print traceback.print_exc()
				true_path = path + ['/'] + ['x'] +[data['path-constraint']]

			for d in data['constraints_candidates']['type_uri']:
				false_path.append(path + ['/'] + ['uri'] + [d])
			for d in data['constraints_candidates']['type_x']:
				false_path.append(path + ['/'] + ['x'] + [d])
			temp['true-path'] = true_path
			temp['false-path'] = false_path
			label_true_path = []
			for token in true_path:
				if token == '/':
					label_true_path.append('/')
				else:
					label_true_path.append(nlutils.tokenize(dbp.get_label(token))[0])
			temp['label-true-path'] = label_true_path
			label_false_paths = []
			for path in false_path:
				temp_fp = []
				for token in path:
					if token == '/':
						temp_fp.append('/')
					else:
						temp_fp.append(nlutils.tokenize(dbp.get_label(token))[0])
				label_false_paths.append(temp_fp)
			temp['label-false-path'] = label_false_paths
			return temp
	# 		false_paths.append(false_path)
	# 		true_paths.append(true_path)
	# 		training_data.append(temp)
	# 		training_data_v2.append([data[u'corrected_question'],data['entity'],label_true_path,label_false_paths,data['_id']])
	# # pickle.dump(training_data,open('resources/training-rdf-data.pickle','w+'))
	# if id:
	# 	idfy(training_data_v2, 'training-rdf-data-v1.pickle')
	# else:
	# 	pickle.dump(training_data_v2,open(DIR+ 'training-rdf-data-v1.pickle','w+'))

def runtime_const(x):

	try:
		data = K.parse_lcquad(x)
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
		data['constraints_candidates'] = {
			"type_x": type_x,
			"type_uri": type_uri
		}
		return data['constraints_candidates']
	except:
		print traceback.print_exc()
		return None


def generate_constraint(data):
	'''
		>This will accept the id_big_data
		>Create relationships with id mapping.
	'''
	#check if the constraint exists
	const = data['parsed-data']['constraints']
	path = ""
	try:
		path = const['?uri']
	except:
		pass
	if not path:
		try:
			path = const['?x']
		except:
			pass
	if path:
		constraints_candidates = runtime_const(data['parsed-data'])
		if not constraints_candidates:
			return None
		else:
			if path not in constraints_candidates['type_uri'] and path not in constraints_candidates['type_x']:
				print "path not in generated candidates"
				return None
			else:
				return constraints_candidates

# strategy_1(main())