'''
	Some scripts to do somethings. I will update it once I write something.
'''
import pickle
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils

short_forms = {
	'dbo:': 'http://dbpedia.org/ontology/',
	'res:': 'http://dbpedia.org/resource/',
	'rdf:': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
	'dbp:': 'http://dbpedia.org/property/'
}



relations_dict = {}
relations_list = []

big_data = pickle.load(open('resources/big_data42.pickle'))
dbp = db_interface.DBPedia(_verbose=True, caching=True)


# data['hop-1-properties']
for data in big_data:
	relations = []
	path = data['parsed-data'][u'path']
	'''
		[u'http://dbpedia.org/property/affiliation', u'http://dbpedia.org/ontology/almaMater']
		This will remove "+" or "-"
	'''
	rel_path = [r[1:] for r in path]
	relations = relations + rel_path
	hop_1 = data['uri']['hop-1-properties']
	hop_1 = [rel[1] for rel in hop_1]
	relations = relations + hop_1
	hop_2 = data['uri']['hop-2-properties']
	for rel in hop_2:
		# temp_rel = [rel[1],rel[3]]
		relations = relations + [rel[1]] + [rel[3]]
	r = list(set(relations))
	relations_list = relations_list + r
	relations_list = list(set(relations_list))

counter = 0
for rel in relations_list:
	'''
		['ID','SF','SF Tokenized','SF ID']
	'''
	surface_form = dbp.get_label(rel)
	surface_form_tokenized = nlutils.tokenize(surface_form)
	surface_form_tokenized_id = embeddings_interface.vocabularize(surface_form_tokenized)
	relations_dict[rel] = [counter,surface_form,surface_form_tokenized,surface_form_tokenized_id]
	counter = counter + 1
pickle.dump(relations_dict,open('resources/relations.pickle','w+'))

id_big_data = []

for data in id_big_data:
