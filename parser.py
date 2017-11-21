"""
    Author: geraltofrivia

    This script takes the json created by the pre-processing module and converts them into X and Y for the network.
    This X and Y depend on the network architecture that we're following so is expected to change now and then.

    Done:
        -> Embed a sentence

"""
import os
import re
import json
import pickle
import warnings
import traceback
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
        WARNING: Give it already split. I ain't splitting it for ye.

        :param _input: The sentence you want embedded. (Assumed pre-tokenized input)
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

        # Small cap everything
        token = token.lower()

        if token == "+":
            token_embedding = np.repeat(1,300)
        elif token == "-":
            token_embedding = np.repeat(-1, 300)
        else:
            try:
                if EMBEDDING == "GLOVE": token_embedding = embedding_glove[token]
                elif EMBEDDING == 'WORD2VEC': token_embedding = embedding_word2vec[token]

            except KeyError:
                if _report_unks: unks.append(token)
                token_embedding = np.zeros(300, dtype=np.float32)

        op += [token_embedding]

    # if DEBUG: print _tokens, "\n",

    return (np.asarray(op), unks) if _report_unks else np.asarray(op)


def tokenize(_input, _ignore_brackets = False):
    """
        Tokenize a question.
        Changes:
            - removes question marks
            - removes commas
            - removes trailing spaces
            - can remove text inside one-level brackets.

        @TODO: Improve tokenization
        :param _input: str, _ignore_brackets: bool
        :return: list of tokens
    """
    cleaner_input = _input.replace("?", "").replace(",", "").strip()
    if _ignore_brackets:
        # If there's some text b/w brackets, remove it. @TODO: NESTED parenthesis not covered.
        pattern = r'\([^\)]*\)'
        matcher = re.search(pattern, cleaner_input, 0)

        if matcher:
            substring = matcher.group()

            cleaner_input = cleaner_input[:cleaner_input.index(substring)] + cleaner_input[cleaner_input.index(substring) + len(substring):]

    return cleaner_input.strip().split()


def compute_true_labels(_question, _truepath, _falsepaths):
    return 1


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

    # Tokenize the question
    question = tokenize(question)

    # Now, embed the question.
    v_question = vectorize(question, False)

    # Make the correct path
    entity = _raw[u'entity'][0]
    entity_sf = tokenize(dbp.get_label(entity), _ignore_brackets=True)   # @TODO: When dealing with many ent, fix this.
    path_sf = []
    for x in _raw[u'path']:
        path_sf.append(x[0])
        path_sf.append(dbp.get_label(x[1:]))
    true_path = entity_sf + path_sf

    """
        Create all possible paths.
        Then choose some of them.
    """
    false_paths = []

    # Collect 1st hop ones.
    for pospath in _raw[u'training'][entity][u'rel1'][0]:
        new_fp = entity_sf + ['+'] + tokenize(dbp.get_label(pospath))
        false_paths.append(new_fp)

    for negpath in _raw[u'training'][entity][u'rel1'][1]:
        new_fp = entity_sf + ['-'] + tokenize(dbp.get_label(negpath))
        false_paths.append(new_fp)


    # Collect 2nd hop ones
    try:

        # Access first element inside rel0 (for paths in direction + )
        for poshop1 in _raw[u'training'][entity][u'rel2'][0]:
            new_fp = entity_sf + ['+']

            hop1 = poshop1.keys()[0]
            hop1sf = dbp.get_label(hop1.replace(",", ""))
            new_fp += tokenize(hop1sf)

            for poshop2 in poshop1[hop1][0]:
                temp_fp = new_fp[:] + ['+'] + tokenize(dbp.get_label(poshop2))
                false_paths.append(temp_fp)

            for neghop2 in poshop1[hop1][1]:
                temp_fp = new_fp[:] + ['-'] + tokenize(dbp.get_label(neghop2))
                false_paths.append(temp_fp)

        # Access second element inside rel0 (for paths in direction - )
        for neghop1 in _raw[u'training'][entity][u'rel2'][1]:
            new_fp = entity_sf + ['-']

            hop1 = neghop1.keys()[0]
            hop1sf = dbp.get_label(hop1.replace(",", ""))
            new_fp += tokenize(hop1sf)

            for poshop2 in neghop1[hop1][0]:
                temp_fp = new_fp[:] + ['+'] + tokenize(dbp.get_label(poshop2))
                false_paths.append(temp_fp)

            for neghop2 in neghop1[hop1][1]:
                temp_fp = new_fp[:] + ['-'] + tokenize(dbp.get_label(neghop2))
                false_paths.append(temp_fp)

    except KeyError:

        # In case there isn't no rel2 in the subgraph, just go with 1 hop paths.
        pass

    # From all these paths, randomly choose some.
    false_paths = np.random.choice(false_paths, MAX_FALSE_PATHS)

    # Vectorize paths
    v_true_path = vectorize(true_path)
    v_false_paths = [ vectorize(x) for x in false_paths ]

    # Corresponding to all these, compute the true labels
    v_y_true = compute_true_labels(question, true_path, false_paths)

    if DEBUG:
        print true_path
        print "now fps"
        print false_paths

    # Throw it out.
    return v_question, v_true_path, v_false_paths, v_y_true


def run(_readfiledir='data/preprocesseddata/', _writefilename='resources/parsed_data.json'):
    """
    Get the show on the road.

    :param _readfiledir:   the filename (directory info included) to read the JSONs that need parsing
    :param _writefilename:  the file to which the parsed (embedded+padded) data is to be written to
    :param _debug:          the boolean param can be overwritten if wanted.
    :return: statuscode(?)
    """

    # Create vars to keep ze data @TODO: think of datatype here
    data_embedded = []

    # Load the vectorizing matrices in memory. TAKES TIME. Prepare your coffee now.
    prepare("GLOVE")

    # Read JSON files.
    for filename in os.listdir(_readfiledir):
        data = json.load(open(os.path.join(_readfiledir, filename)))

        # Each file has multiple datapoints (questions).
        for question in data:

            # Collect the repsonse
            v_q, v_tp, v_fps, v_y = parse(question)

            # Collect data for each question
            data_embedded.append([v_q, v_tp, v_fps, v_y])

    """
        ||TEMP||

        Pickle this data.
        Play around with it.
    """
    f = open('resources/tmp.pickle', 'w+')
    pickle.dump(data_embedded, f)



def test():
    """
        A function to test different things in the script. Will be called from main.

    :return: noting
    """

    """
        Embedding Tests:
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
            embedding = vectorize(tokenize(sentence))
            pprint(embedding)
            raw_input(sentence)

    if "3" in options:
        while True:
            tok = raw_input("Enter your word")
            if tok.lower() in ["stop", "bye", "q", "quit", "exit"]: break
            tok = tok.strip()

            # Quickly manage the multi word problem
            if ' ' in tok:
                tok = tok.split()
            else:
                tok = [tok]

            result = vectorize(tok)
            print result
            try:
                print result.shape
            except AttributeError:
                traceback.print_exc()

    if "4" in options:
        # @TODO: implement this.
        pass


    """
        Parsing tests.
    """
    testdata = json.load(open(os.path.join('data', os.listdir('data')[0])))[-2]
    # data = data.replace("'", '"')
    q, tp, fp, y = parse(testdata)
    pprint(q)
    pprint(tp)
    pprint(fp)
    pprint(y)


if __name__ == "__main__":
    run()