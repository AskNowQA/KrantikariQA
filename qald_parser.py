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

# @TODO: - handle ASK | - handle out of scope stuff | - handle the count

import json
import pickle
import warnings
from pprint import pprint

from utils.dbpedia_interface import DBPedia
from utils import natural_language_utilities as nlutils

# Some macros
DEBUG = True
RAW_QALD_DIR = './resources/qald-7-train-multilingual.json'
PARSED_QALD_DIR = './resources/qald-7-train-parsed.pickle'

# Global variables
dbp = DBPedia(_verbose=True, caching=True)  # Summon a DBpedia interface


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def __fill_single_triple_data__(_triple, _path):
    # Check whether the s or r is the variable
    if str(_triple['subject'][0]) == '?':

        # Template gon' be: e - r
        _entity = [nlutils.is_dbpedia_shorthand(_triple['object'], _convert=True)]
        _path.append('-' + nlutils.is_dbpedia_shorthand(_triple['predicate'], _convert=True))

    elif str(_triple['object'][0]) == '?':

        # Template gon' be: e + r
        _entity = [nlutils.is_dbpedia_shorthand(_triple['subject'], _convert=True)]
        _path.append('+' + nlutils.is_dbpedia_shorthand(_triple['predicate'], _convert=True))

    else:
        warnings.warn("qald_parser:__fill_single_triple_data: Cannot find a variable anywhere. Something forked up")
        return None, None

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

            Returns
                None    : something went wrong
                -1      : out of scope
                [ * ]   : regular stuff

    """
    topic_entities = []
    first_variable = ''

    if not (nlutils.is_dbpedia_uri(_triples[0]['subject']) or nlutils.is_dbpedia_uri(_triples[0]['object'])):
        _triples = [_triples[1], _triples[0]]

    # Okay so now we have a topic entity, lets store it somewhere
    if nlutils.is_dbpedia_uri(_triples[0]['subject']):
        topic_entities, first_variable = [nlutils.is_dbpedia_shorthand(_triples[0]['subject'], _convert=True)], \
                                         _triples[0]['object']
        _path.append('+' + nlutils.is_dbpedia_shorthand(_triples[0]['predicate'], _convert=True))
    elif nlutils.is_dbpedia_uri(_triples[0]['object']):
        topic_entities, first_variable = [nlutils.is_dbpedia_shorthand(_triples[0]['object'], _convert=True)], \
                                         _triples[0]['subject']
        _path.append('-' + nlutils.is_dbpedia_shorthand(_triples[0]['predicate'], _convert=True))
    else:
        warnings.warn("qald_parser.__fill_double_triple_data__: Apparently there is no topic entity in all the SPARQL "
                      + " query. Someone royally forked up. Dying now.")

        """
            For the following SPARQL - we can land upon this condition:
                WHERE {
                    ?uri dbo:office 'President of the United States' .
                    ?uri dbo:orderInOffice '16th' . }

            We just flag it as out of scope and go ahead.
        """
        return -1, -1

    # Based on first_variable, try figuring out the 2nd triple.
    #   either first_v p2 second_v
    #   or first_v p2 ent_2
    #   or second_v p2 first_v
    #   or ent_2 p2 first_v
    # @TODO: verify if I have covered all bases here

    # Check if there an entity in Triple 2
    if nlutils.is_dbpedia_uri(_triples[1]['subject']) or nlutils.is_dbpedia_uri(_triples[1]['object']):

        # There is. Now verify if the other entity is the same as first_variable
        if _triples[1]['subject'] == first_variable:

            # [path] + <pred2>
            topic_entities.append(nlutils.is_dbpedia_shorthand(_triples[1]['object'], _convert=True))
            _path.append('+' + nlutils.is_dbpedia_shorthand(_triples[1]['predicate'], _convert=True))

        elif _triples[1]['object'] == first_variable:

            # [path] - <pred2>
            topic_entities.append(nlutils.is_dbpedia_shorthand(_triples[1]['subject'], _convert=True))
            _path.append('-' + nlutils.is_dbpedia_shorthand(_triples[1]['predicate'], _convert=True))

        else:

            # This makes no sense. In a query with two triples, we can't have two different variables and two entities
            warnings.warn(
                "qald_parser.__fill_double_triple_data__: Apparently there are two topic entities AND two entities "
                + "in this SPARQL query. Someone royally forked up. Dying now.")
            return None, None

    else:

        '''
        There is no entity in the second triple. Then we have two variables.
            - If x rel uri
                - path will be [path] + rel
            - If uri rel x
                - path will be [path] - rel

            ASSUME THAT FIRST VARIABLE IS X *NOT* URI
        '''
        if _triples[1]['subject'] == first_variable:
            _path.append('+' + nlutils.is_dbpedia_shorthand(_triples[1]['predicate'], _convert=True))

        elif _triples[1]['object'] == first_variable:
            _path.append('-' + nlutils.is_dbpedia_shorthand(_triples[1]['predicate'], _convert=True))

        else:
            warnings.warn(
                "qald_parser.__fill_double_triple_data__: Looks like an invalid SPARQL. Returning nones")
            return None, None

    return _path, topic_entities


def get_true_path(sparql):
    """
        Check if there is one or more triples:
            1 Triple:
                not gonna be a rdf:type constraint. Get the sr/ro and make the path.
            2 Triple:
                for every triple
                    do a huge bunch of complicated logic

        Also, if the question has orderby/filterby, do mention that the question is out of scope

    :param sparql:
    :return:
    """
    constraints = {}
    entity = []
    path = []

    # Booleans to make life easy
    has_type_constraint = False
    out_of_scope = False  # @TODO: put in checks for this.

    # Handling keyerror "triples" i.e. there are no triples to start with
    try:
        temp = sparql['where'][0]['triples']
    except KeyError:
        warnings.warn("qald_parser.get_true_path: Cannot find any triple to begin with.")
        return None, None
    finally:
        temp = None

    # Detect and handle ASK questions differently.
    if sparql['queryType'].lower() == 'ask':
        # @TODO: Write this code.
        return None, None

    if len(sparql['where'][0]['triples']) == 1:

        path, entity = __fill_single_triple_data__(_triple=sparql['where'][0]['triples'][0], _path=path)

    elif len(sparql['where'][0]['triples']) == 2:

        '''
            -> Find if there is a type constraint
                -> if so, then on which variable

            -> Assign a topic entity.

        '''

        # Find (if any) the triple with rdf type constraint
        for triple in sparql['where'][0]['triples']:

            if triple['predicate'] in ['a', 'rdf:type', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']:

                has_type_constraint = True
                # Found it. Figure out what is being constrained.
                if triple['subject'] in sparql['variables']:
                    constraints['?uri'] = triple['object']  # The constraint is on the uri
                else:
                    constraints['?x'] = triple['object']

        if has_type_constraint:

            # It means that there is only one triple with real data. That can be taken care of easily.

            for triple in sparql['where'][0]['triples']:
                if not triple['predicate'] in ['a', 'rdf:type', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']:
                    path, entity = __fill_single_triple_data__(_triple=triple, _path=path)

        else:

            # It is a two triple query, but with no rdf:type constraint and we need to parse it the hard way
            path, entity = __fill_double_triple_data__(_triples=sparql['where'][0]['triples'], _path=path)

    elif len(sparql['where'][0]['triples']) == 3:

        '''
            Handle this ONLY if one of the triples is an RDF constraint.

            -> Check if we have an rdf constraint here.
                -> if yes:
                    - parse it and separate it from the triples. Send the rest to __fill_double_triple_data__
        '''
        for triple in sparql['where'][0]['triples']:

            if triple['predicate'] in ['a', 'rdf:type', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']:

                has_type_constraint = True

                # Found it. Figure out what is being constrained.
                if triple['subject'] in sparql['variables']:
                    constraints['?uri'] = triple['object']  # The constraint is on the uri
                else:
                    constraints['?x'] = triple['object']

                # Pop it out of the list of triples and parse the rest
                triples = sparql['where'][0]['triples'][:]
                triples.pop(triples.index(triple))

                path, entity = __fill_double_triple_data__(_triples=triples, _path=path)

        if not has_type_constraint:
            warnings.warn("No code in place for queries with three triples with *NO* rdf:type constraint")
            return None, None

    else:
        warnings.warn("No code in place for queries with more than three triples")
        return None, None

    # Before any return condition, check if anything is None. If so, something somewhere forked up and handle it well.
    return path, entity


def get_false_paths(entity, truepath):
    return None


def run():
    # Load QALD
    raw_dataset = json.load(open(RAW_QALD_DIR))['questions']
    parsed_dataset = pickle.load(open(PARSED_QALD_DIR))

    # Iterate through every question
    for i in range(len(raw_dataset)):

        # Get the QALD question
        q_raw = raw_dataset[i]
        q_parsed = parsed_dataset[i]

        if DEBUG:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(q_raw['query']['sparql'])

        # # Get answer for the query
        # ans = dbp.get_answer(q_raw['query']['sparql'])

        true_path, topic_entities = get_true_path(q_parsed)
        # false_paths = get_false_paths(ans, true_path)

        if DEBUG:
            pprint(true_path)
            pprint(topic_entities)
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # raw_input("Press enter to continue")
        pass


if __name__ == "__main__":
    run()
