'''
	Run this script after completing kranitkari for dataset generation.
'''


import numpy as np
from pprint import pprint
import pickle, json, traceback
from utils import embeddings_interface
from utils import natural_language_utilities as nlutils


MAX_FALSE_PATHS = 1000

def return_combined_result():
	file_name = [250,500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000, 4500, 5000]

	temp_results = []
	temp_parsing_errors = []
	temp_bad_paths = []
	temp_excepts = []

	for file in file_name:
		temp_results.append(pickle.load(open('resources/results' +  str(file) + '.pickle')))
		temp_parsing_errors.append(pickle.load(open('resources/parsing_error' + str(file) + '.pickle')))
		temp_bad_paths.append(pickle.load(open('resources/bad_path' + str(file) + '.pickle')))
		temp_excepts.append(pickle.load(open('resources/except' + str(file) + '.pickle')))

	results = [y for result in temp_results for y in result]
	parsing_errors = [y for result in temp_parsing_errors for y in result]
	bad_paths = [y for result in temp_bad_paths for y in result]
	excepts = [y for result in temp_excepts for y in result]

	data_set = json.load(open('resources/data_set.json'))
	bad_paths_log = []
	for data in data_set:
		for bad_path in bad_paths:
			if data[u'_id'] == bad_path[-1]:
				bad_paths_log.append(data)

	counter = {}
	id = 'sparql_template_id'
	for i in bad_paths_log:
		if i[id] in counter:
			counter[i[id]] = counter[i[id]] + 1
		else:
			counter[i[id]] = 0

	data_set_counter = {}
	id = 'sparql_template_id'
	for i in data_set:
		if i[id] in data_set_counter:
			data_set_counter[i[id]] = data_set_counter[i[id]] + 1
		else:
			data_set_counter[i[id]] = 0


	'''
		Remove all the data points which also belongs to results and also to bad_paths, parsing
	'''
	id_to_remove = []
	for pe in parsing_errors:
		id_to_remove.append(pe[u'_id'])
	for bd in bad_paths_log:
		id_to_remove.append(bd[u'_id'])
	# for e in excepts:
	# 	id_to_remove.append(e[u'_id'])

	new_results = [r for r in results if r[-1] not in id_to_remove]
	return new_results
def main():

	new_results = return_combined_result()
	id_results = []

	counter = 0
	for result in new_results:
		# Id-fy the entire thing
		try:
			id_q = embeddings_interface.vocabularize(nlutils.tokenize(result[0]), _embedding="glove")
			id_tp = embeddings_interface.vocabularize(result[2])
			id_fps = [embeddings_interface.vocabularize(x) for x in result[3]]

			# Actual length of False Paths
			# actual_length_false_path.append(len(id_fps))

			# Makes the number of Negative Samples constant
			id_fps = np.random.choice(id_fps,size=MAX_FALSE_PATHS)

			# Make neat matrices.
			id_results.append([id_q, id_tp, id_fps, np.zeros((20, 1))])
		except:
			'''
				There is some bug in random choice. Need to investigate more on this.
			'''
			counter = counter + 1
	pickle.dump(id_results,open('resources/id_results.pickle','w+'))


def hop_based():
	'''
		This creates a hop based dataset.
		For example Who is the wife of president of America ?
			>initially [q,[president, wife],fps] this gets converted into
			[q , [president], [fp containg only one hop]]
			[q,[president wife],[fp, containg only two hop]
	'''
	results = return_combined_result()
	new_results = []
	for result in results:
		if result[2].count('+') + result[2].count('-') == 1:
			'''
				Remove all the false paths which have more than one hop.
			'''
			_temp_result = []
			_temp_result.append(result[0])	#question
			_temp_result.append(result[1])	#entity
			_temp_result.append(result[2])	#relation/true path
			_temp_false_path = []
			for false_path in result[3]:
				if false_path.count('+') + false_path.count('-') == 1:
					_temp_false_path.append(false_path)
			_temp_result.append(_temp_false_path)
			new_results.append(_temp_result)
		else:
			'''
				Now the true path contains more than one hop. (2)
			'''
			#parsing the true path
			index_location = 0
			if "-" in result[2][1:]:
				index_location = result[2][1:].index('-')
			else:
				index_location = result[2][1:].index('+')

			true_path = result[2][:index_location+1]
			_temp_result = []
			_temp_result.append(result[0])  # question
			_temp_result.append(result[1])  # entity
			_temp_result.append(true_path)  # relation/true path
			_temp_false_path = []
			for false_path in result[3]:
				if false_path.count('+') + false_path.count('-') == 1:
					_temp_false_path.append(false_path)
			try:
				_temp_false_path.remove(true_path)
			except:
				pass
			_temp_result.append(_temp_false_path)
			new_results.append(_temp_result)
			'''
				Generate it for second hop.
			'''
			_temp_result = []
			_temp_result.append(result[0])  # question
			_temp_result.append(result[1])  # entity
			_temp_result.append(result[2])  # relation/true path
			_temp_false_path = []
			for false_path in result[3]:
				if false_path.count('+') + false_path.count('-') != 1:
					_temp_false_path.append(false_path)
			_temp_result.append(_temp_false_path)
			new_results.append(_temp_result)

	id_results = []
	counter = 0

	for result in new_results:
		# Id-fy the entire thing
		try:
			id_q = embeddings_interface.vocabularize(nlutils.tokenize(result[0]), _embedding="glove")
			id_tp = embeddings_interface.vocabularize(result[2])
			id_fps = [embeddings_interface.vocabularize(x) for x in result[3]]

			# Actual length of False Paths
			# actual_length_false_path.append(len(id_fps))

			# Makes the number of Negative Samples constant
			id_fps = np.random.choice(id_fps,size=MAX_FALSE_PATHS)

			# Make neat matrices.
			id_results.append([id_q, id_tp, id_fps, np.zeros((20, 1))])
		except:
			'''
				There is some bug in random choice. Need to investigate more on this.
			'''
			counter = counter + 1
	print counter
	pickle.dump(id_results, open('resources/id_results_hop.pickle', 'w+'))
	return new_results

