import json
from pprint import pprint
import utils.dbpedia_interface as db_interface
import utils.natural_language_utilities as nlutils
import utils.phrase_similarity as sim

short_forms = {
	'dbo:' : 'http://dbpedia.org/ontology/',
	'res:' : 'http://dbpedia.org/resource/',
	'rdf:' : 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
	'dbp:' : 'http://dbpedia.org/property/'
}

#'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'


black_list = ['http://www.w3.org/2000/01/rdf-schema#seeAlso','http://purl.org/linguistics/gold/hypernym',
              'http://www.w3.org/2000/01/rdf-schema#label',
              'http://www.w3.org/2000/01/rdf-schema#comment','http://purl.org/voc/vrank#hasRank','http://xmlns.com/foaf/0.1/isPrimaryTopicOf',
'http://xmlns.com/foaf/0.1/primaryTopic','http://dbpedia.org/ontology/abstract','http://dbpedia.org/ontology/thumbnail',
'http://dbpedia.org/ontology/wikiPageExternalLink','http://dbpedia.org/ontology/wikiPageRevisionID','http://dbpedia.org/ontology/type']

DEBUG = True
dbp = db_interface.DBPedia(_verbose=True, caching=False)

File = 'resources/qald-7-train-multilingual.json'

data = json.load(open(File))
data = data['questions']

data_triple = [] #This is a list of list having [question,[topic_entity],[relation],id]

def get_triples(_sparql_query):
    '''
        parses sparql query to return a set of triples
    '''
    parsed = _sparql_query.split("{")
    parsed = [x.strip() for x in parsed]
    triples = parsed[1][:-1].strip()
    triples =  triples.split(". ")
    triples = [x.strip() for x in triples]
    return triples
#query specific work

def checker(uri,reverse=True,update=True):
	'''
		Checks if uri ends and starts with '>' and '<' respectively. 
		if update= True then also update the uri
	'''
	if uri[0] != '<':
		if update:
			uri = "<" + uri
		else:
			return False
	if uri[-1] != '>':
		if update:
			uri =  uri + ">" 
		else:
			return False
	if reverse:
		return uri[1:-1]
	return uri

def top_k_relation(entity,question,relations,k=20,method=1):
    '''
    :param entity: The entity url in the question
    :param question: The vector form of the question
    :param relations: A list of tuple with the first being the whole relation url and the secodn being incoming or outgoing relation
    :param k: the top k choices
    :return: a list of tuple of relations
    '''
    entity_label = dbp.get_label(entity)
    temp_relations = []
    for rel_tup in relations:
        if rel_tup[0] not in black_list:
            #Find the similarity between the ent+rel and the question
            if method == 1:
            	if rel_tup[0] == 'www.w3.org/1999/02/22-rdf-syntax-ns#type':
            		phrase_1 = entity_label + " " + nlutils.get_label_via_parsing('type')
            	else:		
                	phrase_1 = entity_label + " " + nlutils.get_label_via_parsing(rel_tup[0])
                similarity_score = sim.phrase_similarity(phrase_1, question)
                temp_relations.append((rel_tup[0],rel_tup[1],similarity_score))
            if method == 2:
                phrase_1 = nlutils.get_label_via_parsing(rel_tup[0])
                similarity_score = sim.phrase_similarity(phrase_1, question)
                temp_relations.append((rel_tup[0], rel_tup[1], similarity_score))
        else:
            continue
    temp_relations = sorted(temp_relations, key=lambda tup: tup[2],reverse=True)
    if len(temp_relations) > k:
        return temp_relations[:k]
    else:
		return temp_relations	

for i in xrange(len(data)):
	data[i]['query']['sparql'] = data[i]['query']['sparql'].replace('.\n','. ')
for node in data:
	sparql_query = node['query']['sparql']	#The sparql query of the question
	triples = get_triples(sparql_query) #this will return the all the triples present in the SPARQL query
	triples = [chain.replace(' .','').strip() for chain in triples]	#remove the . from each line of the SPARQL query 
	if len(triples) == 1:
		id = 1 #represents a single triple query
		if "@en" in triples[0]:
			#it has literal. Need to be handeled differently
			continue
		else:
			try:	
				core_chains = triples[0].split(' ')	#split by space to get individual element of chain
				core_chains =[i.strip() for i in core_chains]		#clean up the extra space
				for i in xrange(len(core_chains)):	#replace dbo: with http://dbpedia.org/ontology/ so on and so forth
					for keys in short_forms:
						if keys in core_chains[i]:
							core_chains[i] = core_chains[i].replace(keys,short_forms[keys])
				if "?" in core_chains[0]:
					# implies that the first position is a variable
					# check for '<', '>'
					data_triple.append([node['question'][0]['string'],checker(core_chains[2]),checker(core_chains[1]),id])
				else:
					#implies third position is a variable
					data_triple.append([node['question'][0]['string'],checker(core_chains[0]),checker(core_chains[1]),id])
			except:
				print "haaga"
				continue				
	elif len(triples) == 2:
		pprint(triples)
		temp_core_chains = [chain.split(' ') for chain in triples]
		
	else:
		continue	
'''
	One of the program structure could be - assuming just single entity and single relations (factoid)
	>Take the question
	>Extract the topic entity and also the core chain/s
	>Take top 20 relationships 
	>Match them and find the best and see if it matches the model
	[question,quesion_entity,core relationship]
'''
counter = 0
total = 0
for node in data_triple:
	#vectorize the question
	# vectorized_question = vectorize(node[0])
	try:
		print node[1]
		print node[2]
		temp_right_properties, temp_left_properties = dbp.get_properties(_uri=node[1],label=False)
		#generate possible candidate
		right_properties,left_properties=[],[]
		for properties in temp_right_properties:
			if 'ontology' in properties or 'dbo' in properties:
				right_properties.append(properties)
		for properties in temp_left_properties:
			if 'ontology' in properties or 'dbo' in properties:
				left_properties.append(properties)	
		relations = [(rel,'outgoing') for rel in right_properties]
		relations.extend([(rel,'incoming') for rel in left_properties])
		relations = top_k_relation(node[1],node[0],relations)
		pprint(relations[:5])
		if relations[0][0] == node[2]:
			counter = counter + 1
			print counter
		total = total + 1
		print total
	except:
		print "hagga"
		continue

'''
	Run 1 
In [7]: 48/119
Out[7]: 0

In [8]: 48/119.0
Out[8]: 0.40336134453781514

In [9]: 48/220.0
Out[9]: 0.21818181818181817

	Run 2
	49 -- extebded black list
	Run 3 
	49 -- extended run with rdf#type not being part of the black list

'''