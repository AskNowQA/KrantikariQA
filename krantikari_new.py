'''
	This file heavily depends on krantikari file.

'''

# Imports
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
from utils import model_interpreter
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils

# Some MACROS
DEBUG = True
PATH_CHARS = ['+', '-', '/']
LCQUAD_DIR = './resources/data_set.json'
MAX_FALSE_PATHS = 1000
MODEL_DIR = 'data/training/multi_path_mini/model_00/model.h5'
QALD_DIR = './resources/qald-7-train-multilingual.json'

#CHANGE MACROS HERE
RESULTS_DIR = './resources_v6/results'
LENGTH_DIR = './resources_v6/lengths'
EXCEPT_LOG = './resources_v6/except'
BAD_PATH = './resources_v6/bad_path'
PARSING_ERROR = './resources_v6/parsing_error'
BIG_DATA = './resources_v6/big_data'

short_forms = {
	'dbo:': 'http://dbpedia.org/ontology/',
	'res:': 'http://dbpedia.org/resource/',
	'rdf:': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
	'dbp:': 'http://dbpedia.org/property/'
}

# Load a predicate blacklist from disk
PREDICATE_BLACKLIST = K.PREDICATE_BLACKLIST


class Krantikari_v2:

	def __init__(self, _question, _entities, _dbpedia_interface, _model_interpreter, _qald=False, _training = False, _ask = False ):
		"""
			This function inputs one question, and topic entities, and returns a SPARQL query (or s'thing else)

		:param _question: a string of question
		:param _entities: a list of strings (each being a URI)
		:return: SPARQL/CoreChain/Answers (and or)
		"""
		# QA Specific Macros
		self.K_1HOP_GLOVE = 200 if _training else 20
		self.K_1HOP_MODEL = 5
		self.K_2HOP_GLOVE = 2000 if _training else 10
		self.K_2HOP_MODEL = 5
		self.EMBEDDING = "glove"
		self.TRAINING = _training

		# Training matrix
		self.training_paths = []
		# Stores uri
		self.data = {}

		# Internalize args
		self.question = _question
		self.entities = _entities
		self.qald = _qald
		self.ask = _ask

		# Useful objects
		self.dbp = _dbpedia_interface
		self.model = _model_interpreter

		# @TODO: Catch answers once it returns something.
		self.runtime(self.question, self.entities, self.qald)

	@staticmethod
	def filter_predicates(_predicates, _use_blacklist=True, _only_dbo=False):
		"""
			Function used to filter out predicates based on some logic
				- use a blacklist/whitelist @TODO: Make and plug one in.
				- only use dbo predicates, even.

		:param _predicates: A list of strings (uri of predicates)
		:param _use_blacklist: bool
		:param _only_dbo: bool
		:return: A list of strings (uri)
		"""

		if _use_blacklist:
			_predicates = [x for x in _predicates
						   if x not in PREDICATE_BLACKLIST]

		if _only_dbo:
			_predicates = [x for x in _predicates
						   if x.startswith('http://dbpedia.org/ontology')
						   or x.startswith('dbo:')]

		# Filter out uniques
		_predicates = list(set(_predicates))

		return _predicates

	@staticmethod
	def choose_path_length(hop1_scores, hop2_scores):
		"""
			Function chooses the most probable hop length given hop scores of both 1 and 2 hop scores
			Logic:
				Simply choose which of them have a better score
		:param hop1_scores:
		:param hop2_scores:
		:return: int: Score lenght
		"""
		max_hop1_score = np.max(hop1_scores)
		max_hop2_score = np.max(hop2_scores)

		if max_hop1_score >= max_hop2_score:
			return 1
		else:
			return 2

	@staticmethod
	def get_something(SPARQL, te1, te2, id, dbp):
		if id == 1:
			temp = {}
			temp['te1'] = te1
			temp['te2'] = te2
			answer = dbp.get_answer(SPARQL)  # -,+
			data_temp = []
			for i in xrange(len(answer['r1'])):
				data_temp.append(['-', answer['r1'][i], "+", answer['r2'][i]])
			temp['path'] = data_temp
			return temp
		if id == 2:
			temp = {}
			temp['te1'] = te1
			temp['te2'] = te2
			answer = dbp.get_answer(SPARQL)  # -,+
			data_temp = []
			for i in xrange(len(answer['r1'])):
				data_temp.append(['+', answer['r1'][i], "-", answer['r2'][i]])
			temp['path'] = data_temp
			return temp
		if id == 3:
			temp = {}
			temp['te1'] = te1
			temp['te2'] = te2
			answer = dbp.get_answer(SPARQL)  # -,+
			data_temp = []
			for i in xrange(len(answer['r1'])):
				data_temp.append(['+', answer['r1'][i], "-", answer['r2'][i]])
			temp['path'] = data_temp
			return temp
		if id == 4:
			temp = {}
			temp['te1'] = te1
			temp['te2'] = te2
			answer = dbp.get_answer(SPARQL)  # -,+
			data_temp = []
			for i in xrange(len(answer['r1'])):
				data_temp.append(['+', answer['r1'][i] ])
			temp['path'] = data_temp
			return temp


	@classmethod
	def two_topic_entity(cls, te1, te2, dbp):
		'''
			There are three ways to fit the set of te1,te2 and r1,r2
			 > SELECT DISTINCT ?uri WHERE { ?uri <%(e_to_e_out)s> <%(e_out_1)s> . ?uri <%(e_to_e_out)s> <%(e_out_2)s>}
			 > SELECT DISTINCT ?uri WHERE { <%(e_in_1)s> <%(e_in_to_e_1)s> ?uri. <%(e_in_2)s> <%(e_in_to_e_2)s> ?uri}
			 > SELECT DISTINCT ?uri WHERE { <%(e_in_1)s> <%(e_in_to_e_1)s> ?uri. ?uri <%(e_in_2)s> <%(e_in_to_e_2)s> }
		'''
		te1 = "<" + te1 + ">"
		te2 = "<" + te2 + ">"
		data = []
		SPARQL1 = '''SELECT DISTINCT ?r1 ?r2 WHERE { ?uri ?r1 %(te1)s. ?uri ?r2 %(te2)s . } '''
		SPARQL2 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s ?r1 ?uri.  %(te2)s ?r2 ?uri . } '''
		# SPARQL3 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s ?r1 ?uri.  ?uri ?r2 %(te2)s . } '''

		SPARQL1 = SPARQL1 % {'te1': te1, 'te2': te2}
		SPARQL2 = SPARQL2 % {'te1': te1, 'te2': te2}
		# SPARQL3 = SPARQL3 % {'te1': te1, 'te2': te2}
		data.append(cls.get_something(SPARQL1, te1, te2, 1, dbp))
		# data.append(cls.get_something(SPARQL1, te2, te1, 1, dbp))
		data.append(cls.get_something(SPARQL2, te1, te2, 2, dbp))
		# data.append(cls.get_something(SPARQL2, te2, te1, 2, dbp))
		# data.append(cls.get_something(SPARQL3, te1, te2, 3, dbp))
		# data.append(cls.get_something(SPARQL3, te2, te1, 3, dbp))
		return data

	@classmethod
	def ask_query(cls, te1, te2, dbp):
		te1 = "<" + te1 + ">"
		te2 = "<" + te2 + ">"
		data = []
		SPARQLASK = '''SELECT DISTINCT ?r1  WHERE { %(te1)s ?r1 %(te2)s . } '''
		# SPARQL2 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s ?r1 ?uri.  %(te2)s ?r2 ?uri . } '''
		# SPARQL3 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s ?r1 ?uri.  ?uri ?r2 %(te2)s . } '''

		SPARQL1 = SPARQLASK % {'te1': te1, 'te2': te2}
		# SPARQL2 = SPARQL2 % {'te1': te1, 'te2': te2}
		# SPARQL3 = SPARQL3 % {'te1': te1, 'te2': te2}
		data.append(cls.get_something(SPARQL1, te1, te2, 4, dbp))
		# data.append(cls.get_something(SPARQL1, te2, te1, 1, dbp))
		# data.append(cls.get_something(SPARQL2, te1, te2, 2, dbp))
		# data.append(cls.get_something(SPARQL2, te2, te1, 2, dbp))
		# data.append(cls.get_something(SPARQL3, te1, te2, 3, dbp))
		# data.append(cls.get_something(SPARQL3, te2, te1, 3, dbp))
		return data

	def convert_core_chain_to_sparql(self, _core_chain):  # @TODO
		pass

	def get_hop2_subgraph(self, _entity, _predicate, _right=True):
		"""
			Function fetches the 2hop subgraph around this entity, and following this predicate
		:param _entity: str: URI of the entity around which we need the subgraph
		:param _predicate: str: URI of the predicate with which we curtail the 2hop graph (and get a tangible number of ops)
		:param _right: Boolean: True -> _predicate is on the right of entity (outgoing), else left (incoming)

		:return: List, List: ight (outgoing), left (incoming) preds
		"""
		# Get the entities which will come out of _entity +/- _predicate chain
		intermediate_entities = self.dbp.get_entity(_entity, [_predicate], _right)

		# Filter out the literals, and keep uniques.
		intermediate_entities = list(set([x for x in intermediate_entities if x.startswith('http://dbpedia.org/resource')]))

		if len(intermediate_entities) > 1000:
			intermediate_entities = intermediate_entities[0:1000]
		left_predicates, right_predicates = [], []  # Places to store data.

		for entity in intermediate_entities:
			temp_r, temp_l = self.dbp.get_properties(_uri=entity, label=False)
			left_predicates += temp_l
			right_predicates += temp_r

		return list(set(right_predicates)), list(set(left_predicates))

	def similar_predicates(self, _predicates, _return_indices=False, _k=5):
		"""
			Function used to tokenize the question and compare the tokens with the predicates.
			Then their top k are selected.
		"""
		# If there are no predicates
		if len(_predicates) == 0:
			return np.asarray([]) if _return_indices else []

		# Tokenize question
		qt = nlutils.tokenize(self.question, _remove_stopwords=False)

		# Vectorize question
		v_qt = np.mean(embeddings_interface.vectorize(qt, _embedding=self.EMBEDDING), axis=0)\

		# Declare a similarity array
		similarity_arr = np.zeros(len(_predicates))

		# Fill similarity array
		for i in range(len(_predicates)):
			p = _predicates[i]
			v_p = np.mean(embeddings_interface.vectorize(nlutils.tokenize(p), _embedding=self.EMBEDDING ), axis=0)

			# If either of them is a zero vector, the cosine is 0.\
			if np.sum(v_p) == 0.0 or np.sum(v_qt) == 0.0 or p.strip() == "":
				similarity_arr[i] = np.float64(0.0)
				continue
			try:
				# Cos Product
				similarity_arr[i] = np.dot(v_p, v_qt) / (np.linalg.norm(v_p) * np.linalg.norm(v_qt))
			except:
				traceback.print_exc()


		# Find the best scoring values for every path
		# Sort ( best match score for each predicate) in descending order, and choose top k
		argmaxes = np.argsort(similarity_arr, axis=0)[::-1][:_k]

		if _return_indices:
			return argmaxes

		# Use this to choose from _predicates and return
		return [_predicates[i] for i in argmaxes]

	def runtime(self, _question, _entities, _qald=False):
		'''
			This function inputs one question, and topic entities, and returns a SPARQL query (or s'thing else)

		:param _question: a string of question
		:param _entities: a list of strings (each being a URI)
		:param _qald: bool: Whether or not to use only dbo properties
		:return: SPARQL/CoreChain/Answers (and or)

		data -
			'question-id' : question idified
			data['hop-1-properties'] : [properties] -
					properties are filterd from black list, similarity index and are stored as ['+', http://dbpedia.org/ontology/city]
					#TODO: Verify that there is no '<>'
		'''

		# Two state fail macros
		NO_PATHS = False
		NO_PATHS_HOP2 = False

		# Vectorize the question
		id_q = embeddings_interface.vocabularize(nlutils.tokenize(_question), _embedding=self.EMBEDDING )
		self.data['question-id'] = id_q
		# Algo differs based on whether there's one topic entity or two
		if len(_entities) == 1:

			# Get 1-hop subgraph around the entity
			right_properties, left_properties = self.dbp.get_properties(_uri=_entities[0], label=False)

			# @TODO: Use predicate whitelist/blacklist to trim this shit.
			right_properties = self.filter_predicates(right_properties, _use_blacklist=True, _only_dbo=_qald)
			left_properties = self.filter_predicates(left_properties, _use_blacklist=True, _only_dbo=_qald)

			# Get the surface forms of Entity and the predicates
			entity_sf = self.dbp.get_label(_resource_uri=_entities[0])
			# data['hop-1-properties'] = [right_properties, left_properties]
			#TODO: Verify this logic


			right_properties_sf = [self.dbp.get_label(x) for x in right_properties]
			left_properties_sf = [self.dbp.get_label(x) for x in left_properties]

			# WORD-EMBEDDING FILTERING
			right_properties_filter_indices = self.similar_predicates(_predicates=right_properties_sf,
																	  _return_indices=True,
																	  _k=self.K_1HOP_GLOVE)
			left_properties_filter_indices = self.similar_predicates(_predicates=left_properties_sf,
																	 _return_indices=True,
																	 _k=self.K_1HOP_GLOVE)

			# Impose these indices to generate filtered predicate list.
			right_properties_filtered_sf = [right_properties_sf[i] for i in right_properties_filter_indices]
			left_properties_filtered_sf = [left_properties_sf[i] for i in left_properties_filter_indices]

			# Generate their URI counterparts
			right_properties_filtered_uri = [right_properties[i] for i in right_properties_filter_indices]
			left_properties_filtered_uri = [left_properties[i] for i in left_properties_filter_indices]
			self.data['hop-1-properties'] = [['+', r] for r in right_properties_filtered_uri]
			self.data['hop-1-properties'] += [['-', l] for l in left_properties_filtered_uri]
			self.data['hop-1-properties'] = [list(item) for item in set(tuple(row) for row in self.data['hop-1-properties'])]
			# # Generate 1-hop paths out of them
			# paths_hop1_sf = [nlutils.tokenize(entity_sf, _ignore_brackets=True) + ['+'] + nlutils.tokenize(_p)
			#                  for _p in right_properties_filtered_sf]
			# paths_hop1_sf += [nlutils.tokenize(entity_sf, _ignore_brackets=True) + ['-'] + nlutils.tokenize(_p)
			#                   for _p in left_properties_filtered_sf]

			# Removing entites from the path
			paths_hop1_sf = [['+'] + nlutils.tokenize(_p)
							 for _p in right_properties_filtered_sf]
			paths_hop1_sf += [['-'] + nlutils.tokenize(_p)
							  for _p in left_properties_filtered_sf]

			# Appending the hop 1 paths to the training data (surface forms used)
			self.training_paths += paths_hop1_sf

			# Create their corresponding paths but with URI.
			paths_hop1_uri = [[_entities[0], '+', _p] for _p in right_properties_filtered_uri]
			paths_hop1_uri += [[_entities[0], '-', _p] for _p in left_properties_filtered_uri]





			if not self.TRAINING:
				# Vectorize these paths.
				id_ps = [embeddings_interface.vocabularize(path, _embedding=self.EMBEDDING) for path in paths_hop1_sf]

				# MODEL FILTERING
				hop1_indices, hop1_scores = self.model.rank(_id_q=id_q,
															_id_ps=id_ps,
															_return_only_indices=False,
															_k=self.K_1HOP_MODEL)

				# Impose indices on the paths.
				ranked_paths_hop1_sf = [paths_hop1_sf[i] for i in hop1_indices]
				ranked_paths_hop1_uri = [paths_hop1_uri[i] for i in hop1_indices]

				# if DEBUG:
				#     pprint(ranked_paths_hop1_sf)
				#     pprint(ranked_paths_hop1_uri)

				# Collect URI of predicates so filtered (for 2nd hop)
				left_properties_filtered, right_properties_filtered = [], []

				# Gather all the left and right predicates (from paths selected by the model)
				for i in hop1_indices:

					hop1_path = paths_hop1_sf[i]

					# See if it is from the left or right predicate set.
					if '-' in hop1_path:
						# This belongs to the left pred list.
						# Offset index to match to left_properties_filter_indices index.

						i -= len(right_properties_filter_indices)
						predicate = left_properties[left_properties_filter_indices[i]]
						left_properties_filtered.append(predicate)

					else:
						# This belongs to the right pred list.
						# No offset needed

						predicate = right_properties[right_properties_filter_indices[i]]
						right_properties_filtered.append(predicate)

			else:
				# Create right/left_properties_filtered for training time
				# Collect URI of predicates so filtered (for 2nd hop)
				left_properties_filtered, right_properties_filtered = left_properties_filtered_uri, right_properties_filtered_uri
			"""
				2 - Hop COMMENCES

				Note: Switching to LC-QuAD nomenclature hereon. Refer to /resources/nomenclature.png
			"""
			e_in_in_to_e_in = {}
			e_in_to_e_in_out = {}
			e_out_to_e_out_out = {}
			e_out_in_to_e_out = {}

			for pred in right_properties_filtered:
				temp_r, temp_l = self.get_hop2_subgraph(_entity=_entities[0], _predicate=pred, _right=True)
				e_out_to_e_out_out[pred] = self.filter_predicates(temp_r, _use_blacklist=True, _only_dbo=_qald)
				e_out_in_to_e_out[pred] = self.filter_predicates(temp_l, _use_blacklist=True, _only_dbo=_qald)

			for pred in left_properties_filtered:
				temp_r, temp_l = self.get_hop2_subgraph(_entity=_entities[0], _predicate=pred, _right=False)
				e_in_to_e_in_out[pred] = self.filter_predicates(temp_r, _use_blacklist=True, _only_dbo=_qald)
				e_in_in_to_e_in[pred] = self.filter_predicates(temp_l, _use_blacklist=True, _only_dbo=_qald)

			# Get uniques
			# e_in_in_to_e_in = list(set(e_in_in_to_e_in))
			# e_in_to_e_in_out = list(set(e_in_to_e_in_out))
			# e_out_to_e_out_out = list(set(e_out_to_e_out_out))
			# e_out_in_to_e_out = list(set(e_out_in_to_e_out))

			# Predicates generated. @TODO: Use predicate whitelist/blacklist to trim this shit.

			# if DEBUG:
			#     print("HOP2 Subgraph")
			#     pprint(e_in_in_to_e_in)
			#     pprint(e_in_to_e_in_out)
			#     pprint(e_out_to_e_out_out)
			#     pprint(e_out_in_to_e_out)

			# Get their surface forms, maintain a key-value store
			sf_vocab = {}
			for key in e_in_in_to_e_in.keys():
				for uri in e_in_in_to_e_in[key]:
					sf_vocab[uri] = self.dbp.get_label(uri)
			for key in e_in_to_e_in_out.keys():
				for uri in e_in_to_e_in_out[key]:
					sf_vocab[uri] = self.dbp.get_label(uri)
			for key in e_out_to_e_out_out.keys():
				for uri in e_out_to_e_out_out[key]:
					sf_vocab[uri] = self.dbp.get_label(uri)
			for key in e_out_in_to_e_out.keys():
				for uri in e_out_in_to_e_out[key]:
					sf_vocab[uri] = self.dbp.get_label(uri)

			# Flatten the four kind of predicates, and use their surface forms.
			e_in_in_to_e_in_sf = [sf_vocab[x] for uris in e_in_in_to_e_in.values() for x in uris]
			e_in_to_e_in_out_sf = [sf_vocab[x] for uris in e_in_to_e_in_out.values() for x in uris]
			e_out_to_e_out_out_sf = [sf_vocab[x] for uris in e_out_to_e_out_out.values() for x in uris]
			e_out_in_to_e_out_sf = [sf_vocab[x] for uris in e_out_in_to_e_out.values() for x in uris]

			# WORD-EMBEDDING FILTERING
			e_in_in_to_e_in_filter_indices = self.similar_predicates(_predicates=e_in_in_to_e_in_sf,
																	 _return_indices=True,
																	 _k=self.K_2HOP_GLOVE)
			e_in_to_e_in_out_filter_indices = self.similar_predicates(_predicates=e_in_to_e_in_out_sf,
																	  _return_indices=True,
																	  _k=self.K_2HOP_GLOVE)
			e_out_to_e_out_out_filter_indices = self.similar_predicates(_predicates=e_out_to_e_out_out_sf,
																		_return_indices=True,
																		_k=self.K_2HOP_GLOVE)
			e_out_in_to_e_out_filter_indices = self.similar_predicates(_predicates=e_out_in_to_e_out_sf,
																	   _return_indices=True,
																	   _k=self.K_2HOP_GLOVE)

			# Impose these indices to generate filtered predicate list.
			e_in_in_to_e_in_filtered = [e_in_in_to_e_in_sf[i] for i in e_in_in_to_e_in_filter_indices]
			e_in_to_e_in_out_filtered = [e_in_to_e_in_out_sf[i] for i in e_in_to_e_in_out_filter_indices]
			e_out_to_e_out_out_filtered = [e_out_to_e_out_out_sf[i] for i in e_out_to_e_out_out_filter_indices]
			e_out_in_to_e_out_filtered = [e_out_in_to_e_out_sf[i] for i in e_out_in_to_e_out_filter_indices]

			# Use them to make a filtered dictionary of hop1: [hop2_filtered] pairs
			e_in_in_to_e_in_filtered_subgraph = {}
			for x in e_in_in_to_e_in_filtered:
				for uri in sf_vocab.keys():
					if x == sf_vocab[uri]:

						# That's the URI. Find it's 1-hop Pred.
						for hop1 in e_in_in_to_e_in.keys():
							if uri  in e_in_in_to_e_in[hop1]:

								# Now we found a matching :sweat:
								try:
									e_in_in_to_e_in_filtered_subgraph[hop1].append(uri)
								except KeyError:
									e_in_in_to_e_in_filtered_subgraph[hop1] = [uri]

			e_in_to_e_in_out_filtered_subgraph = {}
			for x in e_in_to_e_in_out_filtered:
				for uri in sf_vocab.keys():
					if x == sf_vocab[uri]:

						# That's the URI. Find it's 1-hop Pred.
						for hop1 in e_in_to_e_in_out.keys():
							if uri in e_in_to_e_in_out[hop1]:

								# Now we found a matching :sweat:
								try:
									e_in_to_e_in_out_filtered_subgraph[hop1].append(uri)
								except KeyError:
									e_in_to_e_in_out_filtered_subgraph[hop1] = [uri]

			e_out_to_e_out_out_filtered_subgraph = {}
			for x in e_out_to_e_out_out_filtered:
				for uri in sf_vocab.keys():
					if x == sf_vocab[uri]:

						if x == 'leader':
							pass

						# That's the URI. Find it's 1-hop Pred.
						for hop1 in e_out_to_e_out_out.keys():
							if uri in e_out_to_e_out_out[hop1]:

								# Now we found a matching :sweat:
								try:
									e_out_to_e_out_out_filtered_subgraph[hop1].append(uri)
								except KeyError:
									e_out_to_e_out_out_filtered_subgraph[hop1] = [uri]

			e_out_in_to_e_out_filtered_subgraph = {}
			for x in e_out_in_to_e_out_filtered:
				for uri in sf_vocab.keys():
					if x == sf_vocab[uri]:

						# That's the URI. Find it's 1-hop Pred.
						for hop1 in e_out_in_to_e_out.keys():
							if uri in e_out_in_to_e_out[hop1]:

								# Now we found a matching :sweat:
								try:
									e_out_in_to_e_out_filtered_subgraph[hop1].append(uri)
								except KeyError:
									e_out_in_to_e_out_filtered_subgraph[hop1] = [uri]

			# Generate 2-hop paths out of them.
			paths_hop2_log = []
			paths_hop2_sf = []
			paths_hop2_uri = []
			for key in e_in_in_to_e_in_filtered_subgraph.keys():
				for r2 in e_in_in_to_e_in_filtered_subgraph[key]:

					# path = nlutils.tokenize(entity_sf, _ignore_brackets=True)   \
					#         + ['-'] + nlutils.tokenize(sf_vocab[key])           \
					#         + ['-'] + nlutils.tokenize(sf_vocab[r2])

					path = ['-'] + nlutils.tokenize(sf_vocab[key]) \
						   + ['-'] + nlutils.tokenize(sf_vocab[r2])
					paths_hop2_sf.append(path)

					path_uri = ['-', key, '-', r2]
					paths_hop2_uri.append(path_uri)

			paths_hop2_log.append(len(paths_hop2_sf))
			for key in e_in_to_e_in_out_filtered_subgraph.keys():
				for r2 in e_in_to_e_in_out_filtered_subgraph[key]:

					# path = nlutils.tokenize(entity_sf, _ignore_brackets=True)   \
					#         + ['-'] + nlutils.tokenize(sf_vocab[key])           \
					#         + ['+'] + nlutils.tokenize(sf_vocab[r2])

					path = ['-'] + nlutils.tokenize(sf_vocab[key]) \
						   + ['+'] + nlutils.tokenize(sf_vocab[r2])
					paths_hop2_sf.append(path)

					path_uri = ['-', key, '+', r2]
					paths_hop2_uri.append(path_uri)

			paths_hop2_log.append(len(paths_hop2_sf))
			for key in e_out_to_e_out_out_filtered_subgraph.keys():
				for r2 in e_out_to_e_out_out_filtered_subgraph[key]:

					# path = nlutils.tokenize(entity_sf, _ignore_brackets=True)   \
					#         + ['+'] + nlutils.tokenize(sf_vocab[key])           \
					#         + ['+'] + nlutils.tokenize(sf_vocab[r2])

					path = ['+'] + nlutils.tokenize(sf_vocab[key]) \
						   + ['+'] + nlutils.tokenize(sf_vocab[r2])

					paths_hop2_sf.append(path)

					path_uri = ['+', key, '+', r2]
					paths_hop2_uri.append(path_uri)

			paths_hop2_log.append(len(paths_hop2_sf))
			for key in e_out_in_to_e_out_filtered_subgraph.keys():
				for r2 in e_out_in_to_e_out_filtered_subgraph[key]:

					# path = nlutils.tokenize(entity_sf, _ignore_brackets=True)   \
					#         + ['+'] + nlutils.tokenize(sf_vocab[key])           \
					#         + ['-'] + nlutils.tokenize(sf_vocab[r2])

					path = ['+'] + nlutils.tokenize(sf_vocab[key]) \
						   + ['-'] + nlutils.tokenize(sf_vocab[r2])
					paths_hop2_sf.append(path)

					path_uri = ['+', key, '-', r2]
					paths_hop2_uri.append(path_uri)

			paths_hop2_log.append(len(paths_hop2_sf))

			# Adding second hop paths to the training set (surface forms)
			# for path in paths_hop2_sf:
			# 	if not path in self.training_paths:
			# 		self.training_paths.append(path)

			self.data['hop-2-properties'] = []

			for path in paths_hop2_sf:
				if not path in self.training_paths:
					self.training_paths.append(path)

			for path in paths_hop2_uri:
				if not path in self.training_paths:
					self.data['hop-2-properties'].append(path)
			#
			# for i in xrange(len(paths_hop2_sf)):
			# 	if not paths_hop2_sf[i] in self.training_paths:
			# 		self.training_paths.append(path)
			# 		self.data['hop-2-properties'].append(paths_hop2_uri[i])



			self.data['hop-2-properties'] = [list(item) for item in
												 set(tuple(row) for row in self.data['hop-2-properties'])]
			#Adding second hop information to the data
			# Run this only if we're not training
			if not self.TRAINING:

				# Vectorize these paths
				id_ps = [embeddings_interface.vocabularize(path, _embedding=self.EMBEDDING) for path in paths_hop2_sf]

				if not len(id_ps) == 0:

					# MODEL FILTERING
					hop2_indices, hop2_scores = self.model.rank(_id_q=id_q,
																_id_ps=id_ps,
																_return_only_indices=False,
																_k=self.K_2HOP_MODEL)

					# Impose indices
					ranked_paths_hop2_sf = [paths_hop2_sf[i] for i in hop2_indices]
					ranked_paths_hop2_uri = [paths_hop2_uri[i] for i in hop2_indices]

					self.path_length = self.choose_path_length(hop1_scores, hop2_scores)

					# @TODO: Merge hop1 and hop2 into one list and then rank/shortlist.

				else:

					# No paths generated at all.
					if DEBUG:
						warnings.warn('No paths generated at the second hop. Question is \"%s\"' % _question)
						warnings.warn('1-hop paths are: \n')
						print(paths_hop1_sf)

					NO_PATHS_HOP2 = True

				# Choose the best path length (1hop/2hop)
				if NO_PATHS_HOP2 is False: self.path_length = self.choose_path_length(hop1_scores, hop2_scores)
			else: self.path_length = 1

		if len(_entities) >= 2:
			self.best_path = 0  # @TODO: FIX THIS ONCE WE IMPLEMENT DIS!
			NO_PATHS = True
			pprint(_entities)
			if not self.ask:
				results = self.two_topic_entity(_entities[0],_entities[1],self.dbp)
			else:
				results = self.ask_query(_entities[0], _entities[1], self.dbp)

			self.data['hop-1-properties'] = []
			self.data['hop-2-properties'] =[]
			final_results = []
			'''
				Filtering out blacklisted relationships and then tokenize and finally vectorized the input.
			'''
			for node in results:
				for paths in node['path']:
					if not (any(x in paths for x in PREDICATE_BLACKLIST)):
						#convert each of the uri to surface forms and then to wordvectors
						_temp_path = []

						for path in paths:
							if path != "+" or path != "-":
								_temp_path.append(nlutils.tokenize(self.dbp.get_label(path)))
							else:
								_temp_path.append([path])
						_temp_path = [y for x in _temp_path for y in x]
						self.data['hop-2-properties'].append(paths)
						#@TODO: Check for two triple about hop-2-properties.
						final_results.append(_temp_path)
			for path in final_results:
				if path not in self.training_paths:
					self.training_paths.append(path)
			self.data['hop-2-properties'] = [list(item) for item in	 set(tuple(row) for row in self.data['hop-2-properties'])]

		# ###########
		# Paths have been generated and ranked.
		#   Now verify the state fail variables and decide what to do
		# ###########

		# If no paths generated, set best path to none
		if NO_PATHS:
			self.best_path = None
			return None

		if not self.TRAINING:

			# Choose best path
			if self.path_length == 1:
				self.best_path = ranked_paths_hop1_uri[np.argmax(hop1_scores)]
			elif self.path_length == 2:
				self.best_path = ranked_paths_hop2_uri[np.argmax(hop2_scores)]

		else:
			self.best_path = np.random.choice(self.training_paths)



def generate_training_data(start,end,qald=False):
	"""
		Function to hack Krantikari to generate model training data.
			- Parse LCQuAD
			- Give it to Kranitkari
			- Collect training paths
			- See if the correct path is not there
			- Append rdf:type constraints to it stochastically # @TODO: This
			- Id-fy the entire thing
			- Make neat matrices (model friendly)
			- Store 'em

	:return:
	"""
	data = []
	bad_path_logs = []
	actual_length_false_path = []
	except_log = []
	big_data = []   #This will store/log everything we need.

	# Create a DBpedia object.
	dbp = db_interface.DBPedia(_verbose=True, caching=True)  # Summon a DBpedia interface

	# Create a model interpreter.
	# model = model_interpreter.ModelInterpreter(_gpu="0")  # Model interpreter to be used for ranking
	model = ""
	if not qald:
		# Load LC-QuAD
		if end == 0:
			dataset = json.load(open(LCQUAD_DIR))[start:]
		else:
			dataset = json.load(open(LCQUAD_DIR))[start:end]
	else:
		# Basic Pre-Processing

		if end == 0:
			dataset = json.load(open(QALD_DIR))['questions'][start:]
		else:
			dataset = json.load(open(QALD_DIR))['questions'][start:end]

		for i in range(len(dataset)):
			dataset[i]['query']['sparql'] = dataset[i]['query']['sparql'].replace('.\n', '. ')

	progbar = ProgressBar()
	iterator = progbar(dataset)
	counter = start


	parsing_error = []
	# Parse it
	for x in iterator:

		try:
			temp_big_data = {}
			if not qald:
				parsed_data = K.parse_lcquad(x)
			else:
				parsed_data = K.parse_qald(x)
			counter = counter + 1
			two_entity = False
			print counter

			if not parsed_data:
				#log this somewhere
				parsing_error.append(x)
				continue

			temp_big_data['parsed-data'] = parsed_data
			'''
				Parsed data would contain triples and the contraints.
			'''
			# Get Needed data
			q = parsed_data[u'corrected_question']
			e = parsed_data[u'entity']
			_id = parsed_data[u'_id']

			if len(e) > 1:
				# results.append([0, 0])
				two_entity = True
			# print q,e
			# Find the correct path
			entity_sf = nlutils.tokenize(dbp.get_label(e[0]), _ignore_brackets=True)  # @TODO: multi-entity alert
			if two_entity:
				entity_sf.append(nlutils.tokenize(dbp.get_label(e[1]), _ignore_brackets=True))

			path_sf = []
			for x in parsed_data[u'path']:
				path_sf.append(str(x[0]))
				path_sf += nlutils.tokenize(dbp.get_label(x[1:]))
			tp = path_sf




			qa = Krantikari_v2(_question=q, _entities=e, _model_interpreter=model, _dbpedia_interface=dbp, _training=True)
			fps = qa.training_paths

			temp_big_data['uri'] = qa.data

			# See if the correct path is there
			try:
				fps.remove(tp)
			except ValueError:

				# The true path was not in the paths generated from Krantikari. Log this son of a bitch.
				if DEBUG:
					print("True path not in false path")
				bad_path_logs.append([q, e, tp, fps,_id])
				continue

			# # Id-fy the entire thing
			# id_q = embeddings_interface.vocabularize(nlutils.tokenize(q), _embedding="glove")
			# id_tp = embeddings_interface.vocabularize(tp)
			# id_fps = [embeddings_interface.vocabularize(x) for x in fps]
			#
			# # Actual length of False Paths
			# actual_length_false_path.append(len(id_fps))
			#
			# # Makes the number of Negative Samples constant
			# id_fps = np.random.choice(id_fps,size=MAX_FALSE_PATHS)
			#
			# # Make neat matrices.
			# data.append([id_q, id_tp, id_fps, np.zeros((20, 1))])

			data.append([q,e,tp,fps,_id])
			temp_big_data['label-data'] = [q,e,tp,fps,_id]
			big_data.append(temp_big_data)
		except Exception:
			except_log.append(x)

			# results.append(evaluate(parsed_data, qa.best_path))

	# I don't know what to do of results. So just pickle shit
	pickle.dump(except_log, open(EXCEPT_LOG, 'w+'))
	pickle.dump(bad_path_logs,open(BAD_PATH,'w+'))
	pickle.dump(data, open(RESULTS_DIR, 'w+'))
	pickle.dump(actual_length_false_path,open(LENGTH_DIR,'w+'))
	pickle.dump(parsing_error,open(PARSING_ERROR,'w+'))
	pickle.dump(big_data,open(BIG_DATA,'w+'))



if __name__ == "__main__":
	# """
	#     TEST1 : Accuracy of similar_predicates
	# """
	# _question = 'Who is the president of Nicaragua ?'
	# p = ['abstract', 'motto', 'population total', 'official language', 'legislature', 'lower house', 'president',
	#      'leader', 'prime minister']
	# _entities = ['http://dbpedia.org/resource/Nicaragua']
	#
	# # Create a DBpedia object.
	# dbp = db_interface.DBPedia(_verbose=True, caching=True)  # Summon a DBpedia interface
	#
	# # Create a model interpreter.
	# model = model_interpreter.ModelInterpreter()  # Model interpreter to be used for ranking
	#
	# qa = Krantikari(_question, _entities,
	#                 _dbpedia_interface=dbp,
	#                 _model_interpreter=model,
	#                 _return_core_chains=True,
	#                 _return_answers=False)
	#
	# print(qa.path_length)

	#
	#
	try:
		append = sys.argv[1]
		start = sys.argv[2]
		end = sys.argv[3]
		qald = sys.argv[4]
	except IndexError:
		# No arguments given. Take from user
		gpu = raw_input("Specify the GPU you wanna use boi:\t")
	#
	# """
	#     TEST 3 : Check generate training data
	# """
	if int(qald) == 0:
		RESULTS_DIR = RESULTS_DIR + append + '.pickle'
		LENGTH_DIR = LENGTH_DIR + append + '.pickle'
		EXCEPT_LOG = EXCEPT_LOG + append + '.pickle'
		BAD_PATH = BAD_PATH + append + '.pickle'
		PARSING_ERROR = PARSING_ERROR + append + '.pickle'
		BIG_DATA = BIG_DATA + append + '.pickle'
		generate_training_data(int(start),int(end),qald=False)
	else:
		RESULTS_DIR = RESULTS_DIR + "qald" + append + '.pickle'
		LENGTH_DIR = LENGTH_DIR + "qald" + append + '.pickle'
		EXCEPT_LOG = EXCEPT_LOG + "qald" + append + '.pickle'
		BAD_PATH = BAD_PATH + "qald" + append + '.pickle'
		PARSING_ERROR = PARSING_ERROR + "qald" + append + '.pickle'
		BIG_DATA = BIG_DATA + append + '.pickle'
		generate_training_data(int(start), int(end), qald=True)