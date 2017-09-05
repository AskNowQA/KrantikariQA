'''
    Filter out all the single triple query pattern.
    Arrange them in a curiculum learning pattern
    Take a entity and make the following json file
    {
        "question"
        "sparql id"
        "entity" : []
    }
'''

import traceback
import json
from pprint import pprint

# Custom files
import utils.dbpedia_interface as db_interface

file_directory = "resources/data_set.json"
json_data = open(file_directory).read()
data = json.loads(json_data)

dbp = db_interface.DBPedia(_verbose=True, caching=False)


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

def get_relationship_hop(_entity, _relation):
    '''
        The objective is to find the outgoing and incoming relationships from the entity at _hop distance.
        :param _entity: the seed entity
        :param _relation: A chain of relation [(rel1,True),(rel2,False)] - True represents a outgoing property while False an incoming property.
        :return: [[set(incoming property)],[set(outgoing property]]
    '''
    for rel in _relation:
        outgoing = rel[1]
        if outgoing:
            ''' get the objects '''





for node in data:
    '''
        For now focusing on just simple question
    '''
    if node[u"sparql_template_id"] == 1:
        '''
        {u'_id': u'9a7523469c8c45b58ec65ed56af6e306',
            u'corrected_question': u'What are the schools whose city is Reading, Berkshire?',
            u'sparql_query': u' SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/ontology/city> <http://dbpedia.org/resource/Reading,_Berkshire> } ',
             u'sparql_template_id': 1,
             u'verbalized_question': u'What are the <schools> whose <city> is <Reading, Berkshire>?'}

        '''

        pprint(node)
        raw_input()
    elif node[u"sparql_template_id"] == 2:
        '''
            {	u'_id': u'8216e5b6033a407191548689994aa32e',
                u'corrected_question': u'Name the municipality of Roberto Clemente Bridge ?',
                u'sparql_query': u' SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Roberto_Clemente_Bridge> <http://dbpedia.org/ontology/municipality> ?uri } ',
                u'sparql_template_id': 2,
                u'verbalized_question': u'What is the <municipality> of Roberto Clemente Bridge ?'
            }
        '''
        data_node = node
        triples = get_triples(node[u'sparql_query'])
        data_node['entity'] = []
        data_node['entity'].append(triples[0].split(" ")[0][1:-1])
        data_node['training'] = {}
        data_node['training'][data_node['entity'][0]] = {}
        data_node['training'][data_node['entity'][0]]['rel1'] =  [list(set(rel)) for rel in list(dbp.get_properties(data_node['entity'][0]))]
        data_node['path'] = ["+" + triples[0].split(" ")[1][1:-1]]
        pprint(data_node)
        raw_input()
