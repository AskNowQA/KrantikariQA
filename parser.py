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

    if DEBUG: print "Question: ", _raw[u'corrected_question']

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

    # Create False paths
    false_paths = []
    if __name__ == '__main__':
        while len(false_paths) < MAX_FALSE_PATHS:
            """
                Logic:
                    -> randomly path length. If the original path has two hops, be more skewed towards more two hops.
                    -> at every hop, randomly select a pos or a neg path.
                        -> case1: equally probable  @TODO: Going with this for now.
                        -> case2: depends upon **number** of predicates in both lists.
                    -> add to the list.

                @TODO: Generalize this for n-hops
            """
            new_fp = [entity_sf]

            path_length = np.random.choice(distributions[1]) if len(_raw[path]) == 1 else np.random.choice(distributions[2])

            # If path_length is one:
            if path_length == 1:
                possible_directions = []
                # Check which direction has possible paths.
                if len(_raw[u'training'][entity][u'rel1'][0]) > 0:
                    possible_directions.append(0)
                if len(_raw[u'training'][entity][u'rel1'][1]) > 0:
                    possible_directions.append(1)

                if not possible_directions:
                    # In both cases, there arent any paths.
                    if DEBUG: print "No more possible paths. Stop making paths for this pred."
                    break

                # Choose between pos/neg paths.
                direction = np.random.choice(possible_directions)     # 0 -> pos; 1 -> neg

                # Pop a random predicate from the rel1 predicates and append it to new fp
                try:
                    selected_pred = _raw[u'training'][entity][u'rel1'][direction].pop(
                        np.random.choice(_raw[u'training'][entity][u'rel1'][direction]))
                # @TODO: At this point, check if the pred has popped from the list or not.
                except IndexError:
                    # Trying to pop from an empty list.
                    break

                # In case the predicate is made up of multiple words, split and add them as different items in the list.
                selected_pred = selected_pred.split()

                # On this last addition (predicate), then apppend the pos/neg sign as decided by the direction.
                # @TODO: This sign can be appended to all the words and not just the first one.
                if direction == 0:
                    selected_pred[0] = '+' + selected_pred[0]
                else:
                    selected_pred[0] = '-' + selected_pred[0]

                # Finally append it to the false path
                if __name__ == '__main__':
                    new_fp.append(selected_pred)

            elif path_length == 2:
                # @TODO: Assuming that the JSON is clean. (No entry with only 1hop pred, and no pred for 2nd hop).

                possible_directions_1hop = []
                if len(_raw[u'training'][entity][u'rel2'][0]) > 0:
                    possible_directions.append(0)
                if len(_raw[u'training'][entity][u'rel1'][1]) > 0:
                    possible_directions.append(1)

                if not possible_directions:
                    # In both cases, there aren't any paths.
                    break

                # Choose between pos/neg paths.
                direction = np.random.choice(possible_directions)     # 0 -> pos; 1 -> neg

                # Select a random predicate from the rel1 preds
                try:
                    selected_pred = np.random.choice(_raw[u'training'][entity][u'rel1'][direction])
                except ValueError:
                    # Trying to choose from an empty list.
                    break

                

                # Break out a secondary path out this as well.
                # In case the predicate is made up of multiple words, split and add them as different items in the list.
                selected_pred = selected_pred.split()

                # On this last addition (predicate), then apppend the pos/neg sign as decided by the direction.
                # @TODO: This sign can be appended to all the words and not just the first one.
                if direction == 0:
                    selected_pred[0] = '+' + selected_pred[0]
                else:
                    selected_pred[0] = '-' + selected_pred[0]

                # Choose b/w +/- on the first hop


    #That's your x.

    # Now to embed this path.


    pass

"""
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


# Prepare X and Y
# @TODO

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
