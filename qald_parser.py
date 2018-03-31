"""

    This script intends to make a BIG_DATA counterpart for QALD questions.
        It expects the SPARQLs to be parsed and stored in JSON format
        via: https://github.com/RubenVerborgh/SPARQL.js

        With the parsed SPARQLs, this will find the true path, and generate their false counterparts as well.

    EXAMPLE of parsed:
    {
        u'distinct': True,
        u'prefixes': {u'dbo': u'http://dbpedia.org/ontology/',
        u'res': u'http://dbpedia.org/resource/'},
        u'queryType': u'SELECT',
        u'type': u'query',
        u'variables': [u'?date'],
        u'where': [{u'triples': [{u'object': u'?date',
                                  u'predicate': u'http://dbpedia.org/ontology/date',
                                  u'subject': u'http://dbpedia.org/resource/Battle_of_Gettysburg'}],
                    u'type': u'bgp'}]}


"""
import json
import pickle

from utils.dbpedia_interface import DBPedia
from utils import natural_language_utilities as nlutils


# Some macros
RAW_QALD_DIR = './resources/qald-7-train-multilingual.json'
PARSED_QALD_DIR = './resources/qald-7-train-parsed.pickle'

# Global variables
dbp = DBPedia(_verbose=True, caching=True)  # Summon a DBpedia interface


def __fill_single_triple_data__(_triple, _path):

    # Check whether the s or r is the variable
    if _triple['subject'] == '?':

        # Template gon' be: e - r
        _entity = nlutils.is_dbpedia_shorthand(_triple['object'], _convert=True)
        _path.append('-' + nlutils.is_dbpedia_shorthand(_triple['predicate'], _convert=True))

    else:

        # Template gon' be: e + r
        _entity = nlutils.is_dbpedia_shorthand(_triple['object'], _convert=True)
        _path.append('-' + nlutils.is_dbpedia_shorthand(_triple['predicate'], _convert=True))

    return _path, _entity


def __fill_double_triple_data__(_triples, _path):
    """
            There is no entity in triple 1
                -> check if there's a topic entity on triple 2 and go ahead with it.

            There is an entity in triple 1
                -> start making path using pred_triple1
                -> stack the variable to find it on triple 2
                -> find the var on triple 2
                    -> no entity there
                        -> set up signs (confusing ones)
                    -> entity there
                        -> chain path (easy peasy)
    """


    #
    # # Find the topic entity
    # for triple in _triples:
    #
    #     if entity: break  # Don't bother if you already have a topic entity
    #
    #     # Check if it is either a shorthand or a complete url (DBpedia url)
    #     if nlutils.is_dbpedia_uri(triple['subject']):
    #
    #         # Declare this to be the topic entity
    #         entity = triple['subject']
    #
    #     elif nlutils.is_dbpedia_uri(triple['object']):
    #
    #         # Declare this to be the topic entity
    #         entity = triple['object']
    #
    #         # We now have a topic entity. Now we want to start creating a path from thereonwards.

    return None, None

def get_true_path(sparql):
    """
        Check if there is one or more triples:
            1 Triple:
                not gonna be a rdf:type constraint. Get the sr/ro and make the path.
            2 Triple:
                for every triple
                    do a huge bunch of complicated logic

    :param sparql:
    :return:
    """
    constraints = {}
    entity = ''
    path = []

    # Booleans to make life easy
    has_constraint = False

    if len(sparql['where']['triples']) == 1:

        path, entity = __fill_single_triple_data__(_triple=sparql['where']['triples'][0], _path=path)

    elif len(sparql['where']['triples']) == 2:

        '''
            -> Find if there is a type constraint
                -> if so, then on which variable

            -> Assign a topic entity.

        '''

        # Find (if any) the triple with rdf type constraint
        for triple in sparql['where']['triples']:

            if triple['predicate'] in ['a', 'rdf:type', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']:

                # Found it. Figure out what is being constrained.
                if triple['subject'] in sparql['variables']:
                    constraints['?uri'] = triple['object']  # The constraint is on the uri
                else:
                    constraints['?x'] = triple['object']

                has_constraint = True

        if has_constraint:

            # It means that there is only one triple with real data. That can be taken care of easily.

            for triple in sparql['where']['triples']:

                if not triple['predicate'] in ['a', 'rdf:type', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']:
                    path, entity = __fill_single_triple_data__(_triple=triple, _path=path)

        else:

            # It is a two triple query, but with no rdf:type constraint and we need to parse it the hard way
            path, entity = __fill_double_triple_data__(_triples=sparql['where']['triples'], _path=path)

    elif len(sparql['where']['triples']) == 3:
        pass

    else:

        pass



def get_false_paths(entity, truepath):

    return None


def run():

    # Load QALD
    raw_dataset = json.load(open(RAW_QALD_DIR))['questions']
    parsed_dataset = pickle.load(open(PARSED_QALD_DIR))

    # Basic Pre-Processing
    raw_dataset = raw_dataset['questions']

    # Iterate through every question
    for i in range(len(raw_dataset)):

        # Get the QALD question
        q_raw = raw_dataset[i]
        q_parsed = raw_dataset[i]

        # # Get answer for the query
        # ans = dbp.get_answer(q_raw['query']['sparql'])

        true_path = get_true_path(q_parsed)
        # false_paths = get_false_paths(ans, true_path)

        pass
