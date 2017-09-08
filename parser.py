"""
    Author: geraltofrivia

    This script takes the json created by the pre-processing module and converts them into X and Y for the network.
    This X and Y depend on the network architecture that we're following so is expected to change now and then.

    Done:
        -> Embed a sentence

"""
import os
import pickle
import warnings
import numpy as np

from pprint import pprint
from gensim import models

from utils import dbpedia_interface as db_interface

# Some Macros and Declarations
DEBUG = True
WORD2VEC_DIR = "./resources"  # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
GLOVE_DIR = "./resources"  # https://nlp.stanford.edu/projects/glove/
EMBEDDING = "GLOVE"  # OR WORD2VEC
EMBEDDING_DIM = 300
MAX_FALSE_PATHS = 20
embedding_glove, embedding_word2vec = {}, {}  # Declaring the two things we're gonna use
distributions = {1: [ 1, 1, 1, 2], 2: [1, 2, 2, 2]}     # p = 3/4 distributions.


# Better warning formatting. Ignore
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = better_warning
if DEBUG: warnings.warn(" DEBUG macro is enabled. Expect cluttered console!")

# Initialize DBpedia
dbp = db_interface.DBPedia(_verbose=True, caching=False)


def prepare(_embedding = EMBEDDING):
    """
        **Call this function prior to doing absolutely anything else.**

        :param _embedding: str | either GLOVE or WORD2VEC.
                Choose which one to use.
        :return: None
    """
    global embedding_glove, embedding_word2vec

    # Preparing embeddings.
    if EMBEDDING == _embedding:

        if DEBUG: print "Using Glove."

        try:
            embedding_glove = pickle.load(open(os.path.join(GLOVE_DIR, "glove_parsed.pickle")))
        except IOError:
            # Glove is not parsed and stored. Do it.
            if DEBUG: warnings.warn(" GloVe is not parsed and stored. This will take some time.")

            embedding_glove = {}
            f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'))
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_glove[word] = coefs
            f.close()

            # Now convert this to a numpy object
            pickle.dump(embedding_glove, open(os.path.join(GLOVE_DIR, "glove_parsed.pickle"), 'w+'))

            if DEBUG: print "GloVe successfully parsed and stored. This won't happen again."

    elif EMBEDDING == _embedding:
        if DEBUG: print "Using Glove."

        embedding_word2vec = models.KeyedVectors.load_word2vec_format(
            os.path.join(WORD2VEC_DIR, 'GoogleNews-vectors-negative300.bin'), binary=True)


def vectorize(_tokens, _report_unks = False):
    """
        Function to embed a sentence and return it as a list of vectors.

        :param _input: The sentence you want embedded.
        :param _report_unks: Whether or not return the out of vocab words
        :return: Numpy tensor of n * 300d, [OPTIONAL] List(str) of tokens out of vocabulary.
    """

    # # Cleaned sentence
    # cleaned_input = _input.replace("?", "").replace(",", "").strip()
    #
    # # Split the sentence into word tokens
    # # @TODO: Use a proper tokenizer.
    # tokens = cleaned_input.split()

    # Logic for Glove
    op = []
    unks = []
    for token in _tokens:
        try:

            #

            if EMBEDDING == "GLOVE": token_embedding = embedding_glove[token]
            elif EMBEDDING == 'WORD2VEC': token_embedding = embedding_word2vec[token]
        except KeyError:
            if _report_unks: unks.append(token)
            token_embedding = np.zeros(300, dtype=np.float32)
        op += [token_embedding]

    if DEBUG: print _tokens, "\n",

    return np.asarray(op) if _report_unks else np.asarray(op), unks


def tokenize(_input):
    """
        Tokenize a question.
        Changes:
            - removes question marks
            - removes commas
            - removes trailing spaces


        :param _input: str
        :return: list of tokens
    """
    return _input.replace("?", "").replace(",", "").strip().split()


def parse(_raw):
    """
    -> parse and vectorize the question.
    -> parse and vectorize the correct path.
    -> create false paths.
        ->
    -> vectorize the false paths.

    :param _raw: A dict. (See ./resources/structure.json)
    :return: None

    @TODO: Generalize this for more than one topic entity.
    """

    # if DEBUG: print "Question: ", str(_raw[u'corrected_question'])

    # Get the question
    question = _raw[u'corrected_question']

    # Now, embed the question.
    ques_vec = vectorize(tokenize(question), False)

    # Make the correct path
    entity = _raw[u'entity'][0]
    entity_sf = dbp.get_label(entity)   # @TODO: When dealing with two entities, fix this.
    path_sf = [x[0]+dbp.get_label(x[1:]) for x in _raw[u'path']]
    path = [entity_sf] + path_sf

    # Now, embed this path.
    path_vec = vectorize(path)

    """
        Create all possible paths and then choose some of them.

        @TODO: Generalize this for n-hops
    """
    false_paths = []

    # Collect 1st hop ones.
    for pospath in _raw[u'training'][entity][u'rel1'][0]:
        new_fp = [entity_sf, '+'] + tokenize(pospath)
        false_paths.append(new_fp)

    for negpath in _raw[u'training'][entity][u'rel1'][1]:
        new_fp = [entity_sf, '-'] + tokenize(negpath)
        false_paths.append(new_fp)

    # Collect 2nd hop ones
    for poshop1 in _raw[u'training'][entity][u'rel2'][0]:
        new_fp = [entity_sf, '+']

        hop1 = poshop1.keys()[0]
        hop1sf = dbp.get_label(hop1)
        new_fp += tokenize(hop1sf)

        for poshop2 in poshop1[hop1][0]:
            temp_fp = new_fp[:] + ['+'] + tokenize(poshop2)
            false_paths.append(temp_fp)

        for neghop2 in poshop1[hop1][1]:
            temp_fp = new_fp[:] + ['-'] + tokenize(neghop2)
            false_paths.append(temp_fp)

    for neghop1 in _raw[u'training'][entity][u'rel2'][0]:
        new_fp = [entity_sf, '-']

        hop1 = neghop1.keys()[0]
        hop1sf = dbp.get_label(hop1)
        new_fp += tokenize(hop1sf)

        for poshop2 in neghop1[hop1][0]:
            temp_fp = new_fp[:] + ['+'] + tokenize(poshop2)
            false_paths.append(temp_fp)

        for neghop2 in neghop1[hop1][1]:
            temp_fp = new_fp[:] + ['-'] + tokenize(neghop2)
            false_paths.append(temp_fp)

    # From all these paths, randomly choose some.
    false_paths = np.random.choice(false_paths, MAX_FALSE_PATHS)

    # @TODO: Test them all.

    # @TODO: Encode them all.
    return path, false_paths



if __name__ == "__main__":
    """
        Embeddding Tests:
            1. Load an embedding
            2. Check it for some sentences
            3. Fire up a loop for token tests
            4. Run it on LC QuAD and report the number of questions with unks and total unks encountered.
    """
    options = raw_input("Type in the numbers corresponding to the tests you want to run.")

    if "1" in options:
        prepare("GLOVE")
        print "Finished loading the embedding. Moving on."

    if "2" in options:
        sents = [
            "Who is the president of United States?",
            "For which band does Joe Hahn perform?",
            "Name some people playing for the Turkish handball league?"
        ]

        for sentence in sents:
            embedding = vectorize(sentence)
            pprint(embedding)
            raw_input(sentence)

    if "3" in options:
        while True:
            tok = raw_input("Enter your word")
            if tok.lower() in ["stop", "bye", "q", "quit", "exit"]: break
            print vectorize(tok)

    if "4" in options:
        # @TODO: implement this.
        pass

    """
        Parsing tests.
    """
    data = """
    Structure of Input JSON.
    {u'_id': u'00a3465694634edc903510572f23b487',
     u'constraints': {},
     u'corrected_question': u'Which party has come in power in Mumbai North?',
     u'entity': [u'http://dbpedia.org/resource/Mumbai_North_(Lok_Sabha_constituency)'],
     u'path': [u'-http://dbpedia.org/property/constituency',
               u'+http://dbpedia.org/ontology/party'],
     u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?x <http://dbpedia.org/property/constituency> <http://dbpedia.org/resource/Mumbai_North_(Lok_Sabha_constituency)> . ?x <http://dbpedia.org/ontology/party> ?uri  . }',
     u'sparql_template_id': 5,
     u'training': {u'http://dbpedia.org/resource/Mumbai_North_(Lok_Sabha_constituency)': {u'rel1': [['wiki Page Uses Template',
                                                                                                     'owl',
                                                                                                     'hypernym',
                                                                                                     '22-rdf-syntax-ns',
                                                                                                     'wiki Page Wiki Link',
                                                                                                     'candidate',
                                                                                                     'prov',
                                                                                                     'wiki Page Wiki Link Text',
                                                                                                     'votes',
                                                                                                     'wiki Page Length',
                                                                                                     'rdf-schema',
                                                                                                     'wiki Page ID',
                                                                                                     'is Primary Topic Of',
                                                                                                     'wiki Page Out Degree',
                                                                                                     'party',
                                                                                                     'percentage',
                                                                                                     'wiki Page Revision ID',
                                                                                                     'abstract',
                                                                                                     'change',
                                                                                                     'subject'],
                                                                                                    ['leaders Seat',
                                                                                                     'blank1 Info Sec',
                                                                                                     'wiki Page Wiki Link',
                                                                                                     'wiki Page Redirects',
                                                                                                     'region',
                                                                                                     'primary Topic',
                                                                                                     'constituency',
                                                                                                     'constituency Mp',
                                                                                                     'is Cited By']],
                                                                                          u'rel2': [[{'http://dbpedia.org/property/party': [[],
                                                                                                                                            []]},
                                                                                                     {'http://dbpedia.org/property/percentage': [[],
                                                                                                                                                 []]}],
                                                                                                    [{'http://dbpedia.org/property/constituency': [['owl',
                                                                                                                                                    'hypernym'],
                                                                                                                                                   ['predecessor',
                                                                                                                                                    'leader Name',
                                                                                                                                                    'is Cited By',
                                                                                                                                                    'wiki Page Wiki Link',
                                                                                                                                                    'before']]},
                                                                                                     {'http://xmlns.com/foaf/0.1/primaryTopic': [[],
                                                                                                                                                 []]},
                                                                                                     {'http://dbpedia.org/property/blank1InfoSec': [['blank2 Info Sec',
                                                                                                                                                     'blank2 Name Sec'],
                                                                                                                                                    ['locale',
                                                                                                                                                     'city',
                                                                                                                                                     'is Cited By',
                                                                                                                                                     'wiki Page Wiki Link',
                                                                                                                                                     'residence']]},
                                                                                                     {'http://dbpedia.org/property/isCitedBy': [[],
                                                                                                                                                []]},
                                                                                                     {'http://dbpedia.org/ontology/wikiPageRedirects': [['wiki Page Uses Template',
                                                                                                                                                         'wiki Page Wiki Link'],
                                                                                                                                                        ['primary Topic']]}]]}},
     u'verbalized_question': u'What is the <party> of the <office holders> whose <constituency> is <Mumbai North (Lok Sabha constituency)>?'}
"""
    tp, fp = parse(data)
    pprint(tp)
    pprint(fp)