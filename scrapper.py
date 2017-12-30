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



def parse_input(node):

	if node[u"sparql_template_id"] in [1, 301, 401, 101] :  # :
		'''
			{
				u'_id': u'9a7523469c8c45b58ec65ed56af6e306',
				u'corrected_question': u'What are the schools whose city is Reading, Berkshire?',
				u'sparql_query': u' SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/ontology/city> <http://dbpedia.org/resource/Reading,_Berkshire> } ',
				u'sparql_template_id': 1,
				u'verbalized_question': u'What are the <schools> whose <city> is <Reading, Berkshire>?'
			}
		'''
		data_node = node
		if ". }" not in node[u'sparql_query']:
			node[u'sparql_query'] = node[u'sparql_query'].replace("}", ". }")
		triples = get_triples(node[u'sparql_query'])
		data_node[u'entity'] = []
		data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
		data_node[u'path'] = ["-" + triples[0].split(" ")[1][1:-1]]
		data_node[u'constraints'] = {}
		if node[u"sparql_template_id"] in [301,401] :
			data_node[u'constraints'] = {triples[1].split(" ")[0]: triples[1].split(" ")[2][1:-1]}
		if node[u"sparql_template_id"] in [401, 101]:
			data_node[u'constraints']['count'] = True
		return data_node

	elif node[u"sparql_template_id"] in [2, 302, 402, 102]:
		'''
			{	u'_id': u'8216e5b6033a407191548689994aa32e',
				u'corrected_question': u'Name the municipality of Roberto Clemente Bridge ?',
				u'sparql_query': u' SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Roberto_Clemente_Bridge> <http://dbpedia.org/ontology/municipality> ?uri } ',
				u'sparql_template_id': 2,
				u'verbalized_question': u'What is the <municipality> of Roberto Clemente Bridge ?'
			}
		'''
		# TODO: Verify the 302 template
		data_node = node
		if ". }" not in node[u'sparql_query']:
			node[u'sparql_query'] = node[u'sparql_query'].replace("}", ". }")
		triples = get_triples(node[u'sparql_query'])
		data_node[u'entity'] = []
		data_node[u'entity'].append(triples[0].split(" ")[0][1:-1])
		data_node[u'path'] = ["+" + triples[0].split(" ")[1][1:-1]]
		data_node[u'constraints'] = {}
		if node[u"sparql_template_id"] in [302,402] :
			data_node[u'constraints'] = {triples[1].split(" ")[0]: triples[1].split(" ")[2][1:-1]}
		if node[u"sparql_template_id"] in [402, 102]:
			data_node[u'constraints']['count'] = True
		return data_node

	elif node[u"sparql_template_id"] in [3, 303, 309, 9, 403, 409, 103, 109]:
		'''
			{    u'_id': u'dad51bf9d0294cac99d176aba17c0241',
				 u'corrected_question': u'Name some leaders of the parent organisation of the Gestapo?',
				 u'sparql_query': u'SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Gestapo> <http://dbpedia.org/ontology/parentOrganisation> ?x . ?x <http://dbpedia.org/ontology/leader> ?uri  . }',
				 u'sparql_template_id': 3,
				 u'verbalized_question': u'What is the <leader> of the <government agency> which is the <parent organisation> of <Gestapo> ?'}
		'''
		# pprint(node)
		data_node = node
		triples = get_triples(node[u'sparql_query'])
		data_node[u'entity'] = []
		data_node[u'entity'].append(triples[0].split(" ")[0][1:-1])
		rel2 = triples[1].split(" ")[1][1:-1]
		rel1 = triples[0].split(" ")[1][1:-1]
		data_node[u'path'] = ["+" + rel1, "+" + rel2]
		data_node[u'constraints'] = {}
		if node[u"sparql_template_id"] in [303, 309, 403, 409]:
			data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[2][1:-1]}
		if node[u"sparql_template_id"] in [403, 409, 103, 109]:
			data_node[u'constraints']['count'] = True
		return data_node

	elif node[u"sparql_template_id"] in [5, 305, 405, 105, 111] :
		'''
			>Verify this !!
			{
				u'_id': u'00a3465694634edc903510572f23b487',
				u'corrected_question': u'Which party has come in power in Mumbai North?',
				u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?x <http://dbpedia.org/property/constituency> <http://dbpedia.org/resource/Mumbai_North_(Lok_Sabha_constituency)> . ?x <http://dbpedia.org/ontology/party> ?uri  . }',
				u'sparql_template_id': 5,
				u'verbalized_question': u'What is the <party> of the <office holders> whose <constituency> is <Mumbai North (Lok Sabha constituency)>?'
			}
		'''
		# pprint(node)
		data_node = node
		triples = get_triples(node[u'sparql_query'])
		rel1 = triples[0].split(" ")[1][1:-1]
		rel2 = triples[1].split(" ")[1][1:-1]
		data_node[u'entity'] = []
		data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
		data_node[u'path'] = ["-" + rel1, "+" + rel2]
		data_node[u'constraints'] = {}
		if node[u"sparql_template_id"] in [305, 405]:
			data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[2][1:-1]}
		if node[u"sparql_template_id"] in [105, 405, 111]:
			data_node[u'constraints']['count'] = True
		return data_node

	elif node[u'sparql_template_id'] in [6, 306, 406, 106]:
		'''
			{
				u'_id': u'd3695db03a5e45ae8906a2527508e7c5',
				u'corrected_question': u'Who have done their PhDs under a National Medal of Science winner?',
				u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?x <http://dbpedia.org/property/prizes> <http://dbpedia.org/resource/National_Medal_of_Science> . ?uri <http://dbpedia.org/property/doctoralAdvisor> ?x  . }',
				u'sparql_template_id': 6,
				u'verbalized_question': u"What are the <scientists> whose <advisor>'s <prizes> is <National Medal of Science>?"
			}
		'''
		# pprint(node)
		data_node = node
		triples = get_triples(node[u'sparql_query'])
		rel1 = triples[0].split(" ")[1][1:-1]
		rel2 = triples[1].split(" ")[1][1:-1]
		data_node[u'entity'] = []
		data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
		data_node[u'path'] = ["-" + rel1, "-" + rel2]
		data_node[u'constraints'] = {}
		if node[u"sparql_template_id"] in [306, 406]:
			data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[2][1:-1]}
		if node[u"sparql_template_id"] in [406, 106]:
			data_node[u'constraints']['count'] = True
		return data_node

	elif node[u'sparql_template_id'] in [7, 8, 307, 308, 407, 408, 107, 108]:
		'''
			{
				u'_id': u'6ff03a568e2e4105b491ab1c1411c1ab',
				u'corrected_question': u'What tv series can be said to be related to the sarah jane adventure and dr who confidential?',
				u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?uri <http://dbpedia.org/ontology/related> <http://dbpedia.org/resource/The_Sarah_Jane_Adventures> . ?uri <http://dbpedia.org/ontology/related> <http://dbpedia.org/resource/Doctor_Who_Confidential> . }',
				u'sparql_template_id': 7,
				u'verbalized_question': u'What is the <television show> whose <relateds> are <The Sarah Jane Adventures> and <Doctor Who Confidential>?'
			 }
		'''
		# pprint(node)
		data_node = node
		triples = get_triples(node[u'sparql_query'])
		rel1 = triples[0].split(" ")[1][1:-1]
		rel2 = triples[1].split(" ")[1][1:-1]
		data_node[u'entity'] = []
		data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
		data_node[u'entity'].append(triples[1].split(" ")[2][1:-1])
		data_node[u'path'] = ["-" + rel1, "+" + rel2]
		data_node[u'constraints'] = {}
		if node[u"sparql_template_id"] in [307, 407, 308, 408]:
			data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[2][1:-1]}
		if node[u"sparql_template_id"] in [407, 107, 408, 108]:
			data_node[u'constraints']['count'] = True
		return data_node

	elif node[u'sparql_template_id'] in [15, 16, 315, 316, 415, 416, 115, 116]:
		'''
			{
				u'_id': u'6ff03a568e2e4105b491ab1c1411c1ab',
				u'corrected_question': u'What tv series can be said to be related to the sarah jane adventure and dr who confidential?',
				u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?uri <http://dbpedia.org/ontology/related> <http://dbpedia.org/resource/The_Sarah_Jane_Adventures> . ?uri <http://dbpedia.org/ontology/related> <http://dbpedia.org/resource/Doctor_Who_Confidential> . }',
				u'sparql_template_id': 7,
				u'verbalized_question': u'What is the <television show> whose <relateds> are <The Sarah Jane Adventures> and <Doctor Who Confidential>?'
			 }
		'''
		data_node = node
		node[u'sparql_query'] = node[u'sparql_query'].replace('uri}', 'uri . }')
		triples = get_triples(node[u'sparql_query'])
		rel1 = triples[0].split(" ")[1][1:-1]
		rel2 = triples[1].split(" ")[1][1:-1]
		data_node[u'entity'] = []
		data_node[u'entity'].append(triples[0].split(" ")[0][1:-1])
		data_node[u'entity'].append(triples[1].split(" ")[0][1:-1])
		data_node[u'path'] = ["+" + rel1, "+" + rel2]
		data_node[u'constraints'] = {}
		if node[u"sparql_template_id"] in [315, 415, 316, 416]:
			data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[2][1:-1]}
		if node[u"sparql_template_id"] in [415, 115, 416, 116]:
			data_node[u'constraints']['count'] = True
		return data_node


for node in temp:
    pprint(parse_input(node))
    raw_input()
