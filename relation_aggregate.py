'''
	Some scripts to do somethings. I will update it once I write something.
'''
import pickle
import os.path
from progressbar import ProgressBar
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils

short_forms = {
	'dbo:': 'http://dbpedia.org/ontology/',
	'res:': 'http://dbpedia.org/resource/',
	'rdf:': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
	'dbp:': 'http://dbpedia.org/property/'
}

dump_location = './resources_v5/'



relations_list = []

print "loading the data from ", dump_location

big_data = pickle.load(open('resources_v5/big_data.pickle'))
dbp = db_interface.DBPedia(_verbose=True, caching=True)

print "done loading the data"

'''
	Check if the 'relations.pickle' location exists.

'''
if os.path.isfile(dump_location + 'relations.pickle'):
	relations_dict = pickle.load(open(dump_location + 'relations.pickle'))
else:
	relations_dict = {}

# data['hop-1-properties']
progbar = ProgressBar()
iterator = progbar(big_data)

for data in iterator:
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

counter = len(relations_dict)

for rel in relations_list:
	'''
		['ID','SF','SF Tokenized','SF ID']
	'''
	if rel not in relations_dict.keys():
		surface_form = dbp.get_label(rel)
		surface_form_tokenized = nlutils.tokenize(surface_form)
		surface_form_tokenized_id = embeddings_interface.vocabularize(surface_form_tokenized)
		relations_dict[rel] = [counter,surface_form,surface_form_tokenized,surface_form_tokenized_id]
		counter = counter + 1

print "dumping the file", dump_location


pickle.dump(relations_dict,open(dump_location + 'relations.pickle','w+'))

print "saving the dump locations"


embeddings_interface.save_out_of_vocab()

print "done saving "


print "idfying things"
id_big_data = []


for data in big_data:
	path_id = [str(p[0])+str(relations_dict[p[1:]][0]) for p in data['parsed-data']['path']]
	data['parsed-data']['path_id'] = path_id
	hop1 = [[r[0], relations_dict[r[1]][0]] for r in data['uri']['hop-1-properties']]
	data['uri']['hop-1-properties'] = hop1
	hop2 = [[r[0], relations_dict[r[1]][0], r[2], relations_dict[r[3]][0]] for r in data['uri']['hop-2-properties']]
	data['uri']['hop-2-properties'] = hop2
	'''
		Remove the true path from the data['uri']. If not found removing the datapoint
	'''
	try:
		if len(path_id) == 1:
			new_path_id = [str(path_id[0]),int(path_id[1:])]
			data['uri']['hop-1-properties'].remove(new_path_id)
		else:
			new_path_id = [str(path_id[0][0]),int(path_id[0][1:]), str(path_id[1][0]),int(path_id[1][1:])]
			data['uri']['hop-2-properties'].remove(new_path_id)
	except:
		continue
	id_big_data.append(data)

print "done with idfying and now saving in the dump location: ", dump_location

pickle.dump(id_big_data,open(dump_location + 'id_big_data.pickle','w+'))

print "done"

