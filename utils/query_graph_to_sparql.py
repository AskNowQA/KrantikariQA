"""
    Bunch of functions that can convert a query graph to SPARQL

"""
import sys
import json
import warnings
import numpy as np

from utils import data_preparation_rdf_type as drt
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils


from utils import  embeddings_interface as ei
ei.__check_prepared__()
vocab = ei.vocab
plus_id = vocab['+']
minus_id = vocab['-']

def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = better_warning

"""
    SPARQL Templates to be used to reconstruct SPARQLs from query graphs
"""
sparql_1hop_template = {
    "-": '%(ask)s %(count)s WHERE { { ?uri <http://dbpedia.org/property/%(r1)s> <%(te1)s> } UNION '
         + '{ ?uri <http://dbpedia.org/ontology/%(r1)s> <%(te1)s> } UNION '
         + '{ ?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s> }. %(rdf)s } ',
    "+": '%(ask)s %(count)s WHERE { { <%(te1)s> <http://dbpedia.org/property/%(r1)s> ?uri } UNION '
         + '{ <%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?uri } UNION'
         + '{ ?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s> } . %(rdf)s }',
    "-c": '%(ask)s %(count)s WHERE { ?uri <%(r1)s> <%(te1)s> . %(rdf)s }',
    "+c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?uri . %(rdf)s }',
}
sparql_boolean_template = {
    "+": '%(ask)s %(count)s WHERE { { <%(te1)s> <http://dbpedia.org/property/%(r1)s> <%(te2)s> } UNION '
         + '{ <%(te1)s> <http://dbpedia.org/ontology/%(r1)s> <%(te2)s> } UNION '
         + '{ ?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s> } . %(rdf)s }',
    "+s": '%(ask)s %(count)s WHERE { { <%(te1)s> <http://dbpedia.org/property/%(r1)s> ?uri } UNION '
         + '{ <%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?uri } UNION '
         + '{ ?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s> } . %(rdf)s }',
    "-s": '%(ask)s %(count)s WHERE { { ?uri <http://dbpedia.org/property/%(r1)s> <%(te1)s> } UNION '
         + '{ ?uri <http://dbpedia.org/ontology/%(r1)s> <%(te1)s> } UNION '
         + '{ ?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s> } . %(rdf)s }'
    # "": '%(ask)s %(count)s WHERE { { <%(te1)s> <http://dbpedia.org/property/%(r1)s> <%(te2)s> } UNION '
    #                              + '{ <%(te1)s> <http://dbpedia.org/ontology/%(r1)s> <%(te2)s> } . %(rdf)s }',
}
sparql_2hop_1ent_template = {
    "++": '%(ask)s %(count)s WHERE { {<%(te1)s> <http://dbpedia.org/property/%(r1)s> ?x} UNION '
          + '{<%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?x} UNION '
          + '{<%(te1)s> <http://purl.org/dc/terms/%(r1)s> ?x} . '
          + '{?x <http://dbpedia.org/property/%(r2)s> ?uri} UNION'
          + '{?x <http://dbpedia.org/ontology/%(r2)s> ?uri} UNION'
          + '{?x <http://purl.org/dc/terms/%(r2)s> ?uri} . %(rdf)s }',
    "-+": '%(ask)s %(count)s WHERE { {?x <http://dbpedia.org/property/%(r1)s> <%(te1)s>} UNION '
          + '{?x <http://dbpedia.org/ontology/%(r1)s> <%(te1)s>} UNION '
          + '{?x <http://purl.org/dc/terms/%(r1)s> <%(te1)s>} .'
          + '{?x <http://dbpedia.org/property/%(r2)s> ?uri} UNION '
          + '{?x <http://dbpedia.org/ontology/%(r2)s> ?uri} UNION '
          + '{?x <http://purl.org/dc/terms/%(r2)s> ?uri} . %(rdf)s }',
    "--": '%(ask)s %(count)s WHERE { {?x <http://dbpedia.org/property/%(r1)s> <%(te1)s>} UNION '
          + '{?x <http://dbpedia.org/ontology/%(r1)s> <%(te1)s>} UNION '
          + '{?x <http://purl.org/dc/terms/%(r1)s> <%(te1)s>} .'
          + '{?uri <http://dbpedia.org/property/%(r2)s> ?x} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> ?x} UNION '
          + '{?uri <http://purl.org/dc/terms/%(r2)s> ?x} . %(rdf)s }',
    "+-": '%(ask)s %(count)s WHERE { {<%(te1)s> <http://dbpedia.org/property/%(r1)s> ?x} UNION '
          + '{<%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?x} UNION '
          + '{<%(te1)s> <http://purl.org/dc/terms/%(r1)s> ?x} .'
          + '{?uri <http://dbpedia.org/property/%(r2)s> ?x} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> ?x} UNION '
          + '{?uri <http://purl.org/dc/terms/%(r2)s> ?x} . %(rdf)s }',
    "++c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?x . ?x <%(r2)s> ?uri . %(rdf)s }',
    "-+c": '%(ask)s %(count)s WHERE { ?x <%(r1)s> <%(te1)s> . ?x <%(r2)s> ?uri . %(rdf)s }',
    "--c": '%(ask)s %(count)s WHERE { ?x <%(r1)s> <%(te1)s> . ?uri <%(r2)s> ?x . %(rdf)s }',
    "+-c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?x . ?uri <%(r2)s> ?x . %(rdf)s }'
}

sparql_2hop_2ent_template = {
    "+-": '%(ask)s %(count)s WHERE { {<%(te1)s> <http://dbpedia.org/property/%(r1)s> ?uri} UNION '
          + '{<%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?uri} UNION '
          + '{<%(te1)s> <http://purl.org/dc/terms/%(r1)s> ?uri} . '
          + '{<%(te2)s> <http://dbpedia.org/property/%(r2)s> ?uri} UNION '
          + '{<%(te2)s> <http://dbpedia.org/ontology/%(r2)s> ?uri} UNION'
          + '{<%(te2)s> <http://purl.org/dc/terms/%(r2)s> ?uri} . %(rdf)s }',
    "--": '%(ask)s %(count)s WHERE { {?uri <http://dbpedia.org/property/%(r1)s> <%(te1)s>} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r1)s> <%(te1)s>} UNION '
          + '{?uri <http://purl.org/dc/terms/s> <%(te1)s>} . '
          + '{?uri <http://dbpedia.org/property/%(r2)s> <%(te2)s>} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> <%(te2)s>} UNION '
          + '{?uri <http://purl.org/dc/terms/%(r2)s> <%(te2)s>} . %(rdf)s }',
    "-+": '%(ask)s %(count)s WHERE { {?uri <http://dbpedia.org/property/%(r1)s> <%(te1)s>} UNION '
          + '{?uri <http://dbpedia.org/ontology/%(r1)s> <%(te1)s>} UNION'
          + '{?uri <http://purl.org/dc/terms/%(r1)s> <%(te1)s>} .'
          + '{?uri <http://dbpedia.org/property/%(r2)s> <%(te2)s>} UNION'
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> <%(te2)s>} UNION'
          + '{?uri <http://purl.org/dc/terms/%(r2)s> <%(te2)s>} . %(rdf)s }',
    "++": '%(ask)s %(count)s WHERE { {<%(te1)s> <http://dbpedia.org/property/%(r1)s> ?uri} UNION '
          + '{<%(te1)s> <http://dbpedia.org/ontology/%(r1)s> ?uri} UNION'
          + '{<%(te1)s> <http://purl.org/dc/terms/%(r1)s> ?uri} .'
          + '{?uri <http://dbpedia.org/property/%(r2)s> <%(te2)s>} UNION'
          + '{?uri <http://dbpedia.org/ontology/%(r2)s> <%(te2)s>} UNION'
          + '{?uri <http://purl.org/dc/terms/%(r2)s> <%(te2)s>}  . %(rdf)s }',
    "+-c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?uri . <%(te2)s> <%(r2)s> ?uri . %(rdf)s }',
    "--c": '%(ask)s %(count)s WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <%(te2)s> . %(rdf)s }',
    "-+c": '%(ask)s %(count)s WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <%(te2)s> . %(rdf)s }',
    "++c": '%(ask)s %(count)s WHERE { <%(te1)s> <%(r1)s> ?uri . ?uri <%(r2)s> <%(te2)s> . %(rdf)s }',
}

rdf_constraint_template = ' ?%(var)s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <%(uri)s> . '
RDF_TYPE_LOOKUP_LOC = 'data/data/common/rdf_type_lookup.json'

# Globals to be initialized externally
dbp, reverse_rdf_type = None, None


def return_sign(sign):
    if sign == plus_id:
        return '+'
    else:
        return '-'


def rdf_type_candidates(data, path_id, relations):
    '''
        Takes in path ID (continous IDs, not glove vocab).
        And return type candidates (in continous IDs)
            based on whether we want URI or X candidates
    :param data:
    :param path_id:
    :param relations:
    :param reverse_vocab:
    :param core_chain:
    :return:
    '''

    # @TODO: Only generate specific candidates
    data = data['parsed-data']
    path = id_to_path(path_id, relations, core_chain=True)
    sparql = drt.reconstruct(data['entity'], path, alternative=True)
    sparqls = drt.create_sparql_constraints(sparql)


    if len(data['entity']) == 2 or len(path) == 2:
        sparqls = [sparqls[1]]
    type_x, type_uri = drt.retrive_answers(sparqls)

    type_x_candidates, type_uri_candidates = drt.create_valid_paths(type_x, type_uri)

    # Convert them to continous IDs.
    for i in range(len(type_x_candidates)):
        for j in range(len(type_x_candidates[i])):
            try:
                type_x_candidates[i][j] = type_x_candidates[i][j]
            except KeyError:
                '''
                    vocab[1] refers to unknow word.
                '''
                type_x_candidates[i][j] = [1]
    for i in range(len(type_uri_candidates)):
        for j in range(len(type_uri_candidates[i])):
            try:
                type_uri_candidates[i][j] = type_uri_candidates[i][j]
            except:
                type_uri_candidates[i][j] = [1]

    # Return based on given input.
    return type_x_candidates + type_uri_candidates


def id_to_path(path_id, relations, core_chain = True):
    '''


    :param path_id:  array([   3, 3106,    3,  647]) - corechain wihtout entity
    :param vocab: from continuous id space to discrete id space.
    :param relations: inverse relation lookup dictionary
    :return: paths
    '''

    # mapping from discrete space to continuous space.
    # path_id = np.asarray([embeddingid_to_gloveid[i] for i in path_id])

    # find all the relations in the given paths
    if core_chain:
        '''
            Identify the length. Is it one hop or two.
            The assumption is '+' is 2 and '-' is 3
        '''
        rel_length = 1
        if plus_id in path_id[1:].tolist() or minus_id in path_id[1:].tolist():
            rel_length = 2

        if rel_length == 2:
            sign_1 = path_id[0]
            try:
                index_sign_2 = path_id[1:].tolist().index(plus_id) + 1
            except ValueError:
                index_sign_2 = path_id[1:].tolist().index(minus_id) + 1
            rel_1 ,rel_2 = path_id[1:index_sign_2] ,path_id[index_sign_2 +1:]
            rel_1 = rel_id_to_rel(rel_1 ,relations)
            rel_2 = rel_id_to_rel(rel_2 ,relations)
            sign_2 = path_id[index_sign_2]
            path = [return_sign(sign_1) ,rel_1 ,return_sign(sign_2) ,rel_2]
            return path
        else:
            sign_1 = path_id[0]
            rel_1 = path_id[1:]
            rel_1 = rel_id_to_rel(rel_1 ,relations)
            path = [return_sign(sign_1) ,rel_1]
            return path
    else:
        variable = path_id[0]
        sign_1 = path_id[1]
        rel_1 = rel_id_to_rel(path_id[2:] ,relations)
        pass


def load_reverse_rdf_type(embeddings_interface):
    if sys.version_info[0] == 3:
        rdf_type = json.load(open(RDF_TYPE_LOOKUP_LOC,'rb'))
    rdf = {}
    for classes in rdf_type:
        rdf[classes] = embeddings_interface.vocabularize(nlutils.tokenize(dbp.get_label(classes)))
    return rdf


def rel_id_to_rel(rel, _relations):
    """


    :param rel:
    :param _relations: The relation lookup is inverse here
    :return:
    """
    occurrences = []
    for key in _relations:
        value = _relations[key]
        if np.array_equal(value[3] ,np.asarray(rel)):
            occurrences.append(value)
    # print occurrences
    if len(occurrences) == 1:
        return occurrences[0][0]
    else:
        '''
            prefers /dc/terms/subject' and then ontology over properties
        '''
        if 'terms/subject' in occurrences[0][0]:
            return occurrences[0][0]
        if 'terms/subject' in occurrences[1][0]:
            return occurrences[1][0]
        if 'property' in occurrences[0][0]:
            return occurrences[1][0]
        else:
            return occurrences[0][0]

plus_id, minus_id = None, None # These are vocab IDs
def reconstruct_corechain(_chain ,relations, embeddings_interface):
    """
        Expects a corechain made of continous IDs, and returns it in its text format (uri form)
        @TODO: TEST!
    :param _chain: list of ints
    :return: str: list of strs
    """
    global plus_id, minus_id

    # Find the plus and minus ID.
    if not (plus_id and minus_id):
        plus_id = embeddings_interface.vocabularize(['+'])
        minus_id = embeddings_interface.vocabularize(['-'])

    # corechain_vocabbed = [embeddingid_to_gloveid[key] for key in _chain]
    corechain_vocabbed = _chain
    # Find the hop-length of the corechain
    length = sum([ 1 for id in corechain_vocabbed if id in [plus_id, minus_id]])

    if length == 1:

        # Just one predicate. Find its uri
        uri = rel_id_to_rel(corechain_vocabbed[1:] ,relations)
        sign = '+' if corechain_vocabbed[0] == plus_id else '-'
        signs = [sign]
        uris = [uri]

    elif length == 2:

        # Find the index of the second sign
        index_second_sign = None
        for i in range(1, len(corechain_vocabbed)):
            if corechain_vocabbed[i] in [plus_id, minus_id]:
                index_second_sign = i
                break

        first_sign = '+' if corechain_vocabbed[0] == plus_id else '-'
        second_sign = '+' if corechain_vocabbed[index_second_sign] == plus_id else '-'
        first_uri = rel_id_to_rel(corechain_vocabbed[1:index_second_sign] ,relations)
        second_uri = rel_id_to_rel(corechain_vocabbed[index_second_sign +1:] ,relations)

        signs = [first_sign, second_sign]
        uris = [first_uri, second_uri]

    else:
        # warnings.warn("Corechain length is unexpected. Help!")
        return [], []

    return signs, uris


def convert_rdf_path_to_text(path, embeddings_interface):
    """
        Function used to convert a path relations(of continous IDs) to a text path.
        Eg. [ 5, 3, 420] : [uri, dbo:Poop]

    :param path: list of strings
    :return: list of text
    """

    # First we need to convert path to glove vocab
    # path = [embeddingid_to_gloveid[x] for x in path]

    # Then to convert this to text
    var = ''
    for key in embeddings_interface.vocab.keys():
        if embeddings_interface.vocab[key] == path[0]:
            var = key
            break

    dbo_class = ''
    for key in reverse_rdf_dict.keys():
        if list(reverse_rdf_dict[key]) == list(path[2:]):
            dbo_class = key
            break

    return [var, dbo_class]


def convert(_graph, relations, embeddings_interface):
    """
        Expects a dict containing:
            best_path,
            intent,
            rdf_constraint,
            rdf_constraint_type,
            rdf_best_path

        Returns a composted SPARQL.

        1. Convert everything to strings.

    :param _graph: (see above)
    :return: str: SPARQL.
    """
    sparql_value = {}

    # Find entities
    entities = _graph['entities']

    # print _graph

    # Convert the corechain to glove embeddings
    corechain_signs, corechain_uris = reconstruct_corechain(_graph['best_path'],
                                                            relations,
                                                            embeddings_interface)

    # Construct the stuff outside the where clause
    sparql_value["ask"] = 'ASK' if _graph['intent'] == 'ask' else 'SELECT DISTINCT'
    if _graph['intent'] == 'count':
        sparql_value["count"] = 'COUNT(?uri)'
    elif _graph['intent'] == 'ask':
        sparql_value["count"] = ''
    else:
        sparql_value["count"] = '?uri'

    # Check if we need an RDF constraint.
    if _graph['rdf_constraint']:
        try:
            rdf_constraint_values = {}
            rdf_constraint_values['var'] = _graph['rdf_constraint_type']
            rdf_constraint_values['uri'] = convert_rdf_path_to_text(_graph['rdf_best_path'],
                                                                    embeddings_interface=embeddings_interface)[1]

            sparql_value["rdf"] = rdf_constraint_template % rdf_constraint_values
        except IndexError:
            sparql_value["rdf"] = ''

    else:
        sparql_value["rdf"] = ''

    # Find the particular template based on the signs

    """
        Start putting stuff in template.
        Note: if we're dealing with a count query, we append a 'c' to the query.
            This does away with dbo/dbp union and goes ahead with whatever came in the question.
            
        Note: In the case where its a 2hop query, with ask intent:
            we only consider the first sign, an ignore the second one. Can lead to incorrect queries.
    """
    signs_key = ''.join(corechain_signs)

    if _graph['intent'] == 'ask':
        # Assuming that there is only single triple ASK queries.
        sparql_value["te1"] = _graph['entities'][0]
        try:
            sparql_value["te2"] = _graph['entities'][1]
            sparql_template = sparql_boolean_template['+']
        except IndexError:
            warnings.warn("Found a single entity boolean question")
            sparql_template = sparql_boolean_template[signs_key[0] +'s']

        sparql_value["r1"] = corechain_uris[0].split('/')[-1]

    elif len(signs_key) == 1:
        print("DEBUG:  ", signs_key)
        # Single hop, non boolean.
        sparql_template = sparql_1hop_template[signs_key +'c' if _graph['intent'] == 'count' else signs_key]
        sparql_value["te1"] = _graph['entities'][0]
        sparql_value["r1"] = corechain_uris[0] if _graph['intent'] == 'count' else corechain_uris[0].split('/')[-1]

    else:
        # Double hop, non boolean.

        sparql_value["r1"] = corechain_uris[0] if _graph['intent'] == 'count' else corechain_uris[0].split('/')[-1]
        sparql_value["r2"] = corechain_uris[1] if _graph['intent'] == 'count' else corechain_uris[1].split('/')[-1]

        # Check if entities are one or two
        if len(_graph['entities']) == 1:
            sparql_template = sparql_2hop_1ent_template[signs_key +'c' if _graph['intent'] == 'count' else signs_key]
            sparql_value["te1"] = _graph['entities'][0]
        else:
            sparql_template = sparql_2hop_2ent_template[signs_key +'c' if _graph['intent'] == 'count' else signs_key]
            sparql_value["te1"] = _graph['entities'][0]
            sparql_value["te2"] = _graph['entities'][1]

    # Now to put the magic together
    sparql = sparql_template % sparql_value

    return sparql


def convert_runtime(_graph):
    """
        Expects a dict containing:
            best_path,
            intent,
            rdf_constraint,
            rdf_constraint_type,
            rdf_best_path

        Returns a composted SPARQL.

        1. Convert everything to strings.

    :param _graph: (see above)
    :return: str: SPARQL.
    """
    sparql_value = {}

    # Find entities
    entities = _graph['entities']
    best_path = _graph['best_path']


    if len(best_path) == 2:
        corechain_signs  = [best_path[0]]
        corechain_uris = [best_path[1]]
    else:
        corechain_signs = [best_path[0],best_path[2]]
        corechain_uris = [best_path[1],best_path[3]]



    # Construct the stuff outside the where clause
    sparql_value["ask"] = 'ASK' if _graph['intent'] == 'ask' else 'SELECT DISTINCT'
    if _graph['intent'] == 'count':
        sparql_value["count"] = 'COUNT(?uri)'
    elif _graph['intent'] == 'ask':
        sparql_value["count"] = ''
    else:
        sparql_value["count"] = '?uri'

    # Check if we need an RDF constraint.
    if _graph['rdf_constraint']:
        try:
            rdf_constraint_values = {}
            rdf_constraint_values['var'] = _graph['rdf_constraint_type']
            rdf_constraint_values['uri'] = _graph['rdf_best_path']

            sparql_value["rdf"] = rdf_constraint_template % rdf_constraint_values
        except IndexError:
            sparql_value["rdf"] = ''

    else:
        sparql_value["rdf"] = ''

    # Find the particular template based on the signs

    """
        Start putting stuff in template.
        Note: if we're dealing with a count query, we append a 'c' to the query.
            This does away with dbo/dbp union and goes ahead with whatever came in the question.

        Note: In the case where its a 2hop query, with ask intent:
            we only consider the first sign, an ignore the second one. Can lead to incorrect queries.
    """
    signs_key = ''.join(corechain_signs)

    if _graph['intent'] == 'ask':
        # Assuming that there is only single triple ASK queries.
        sparql_value["te1"] = _graph['entities'][0]
        try:
            sparql_value["te2"] = _graph['entities'][1]
            sparql_template = sparql_boolean_template['+']
        except IndexError:
            warnings.warn("Found a single entity boolean question")
            sparql_template = sparql_boolean_template[signs_key[0] + 's']

        sparql_value["r1"] = corechain_uris[0].split('/')[-1]

    elif len(signs_key) == 1:
        print("DEBUG:  ", signs_key)
        # Single hop, non boolean.
        sparql_template = sparql_1hop_template[signs_key + 'c' if _graph['intent'] == 'count' else signs_key]
        sparql_value["te1"] = _graph['entities'][0]
        sparql_value["r1"] = corechain_uris[0] if _graph['intent'] == 'count' else corechain_uris[0].split('/')[-1]

    else:
        # Double hop, non boolean.

        sparql_value["r1"] = corechain_uris[0] if _graph['intent'] == 'count' else corechain_uris[0].split('/')[-1]
        sparql_value["r2"] = corechain_uris[1] if _graph['intent'] == 'count' else corechain_uris[1].split('/')[-1]

        # Check if entities are one or two
        if len(_graph['entities']) == 1:
            sparql_template = sparql_2hop_1ent_template[signs_key + 'c' if _graph['intent'] == 'count' else signs_key]
            sparql_value["te1"] = _graph['entities'][0]
        else:
            sparql_template = sparql_2hop_2ent_template[signs_key + 'c' if _graph['intent'] == 'count' else signs_key]
            sparql_value["te1"] = _graph['entities'][0]
            sparql_value["te2"] = _graph['entities'][1]

    # Now to put the magic together
    sparql = sparql_template % sparql_value

    return sparql



def init(embedding_interface):
    global dbp, reverse_rdf_dict

    dbp = db_interface.DBPedia(_verbose=False, caching=True)
    reverse_rdf_dict = load_reverse_rdf_type(embedding_interface)