"""
    Script that runs an embedding server and is expected to be left alone in the RAM
        NOTE: DO NOT CALL THIS FILE, IT IS TO BE CALLED BY AN EMBEDDING INTERFACE.

"""

import os
import json
import gensim
import pickle
import bottle
import warnings
import numpy as np

from bottle import post, get, put, delete, request, response

word2vec_embeddings = None
glove_embeddings = None
DEFAULT_EMBEDDING = 'word2vec'
DEBUG = True
PORT = 6969
glove_location = \
    {
        'dir': "./resources",
        'raw': "glove.6B.300d.txt",
        'parsed': "glove_parsed_small.pickle"
    }


def start():
    """
        Call this function to
            - start the server if not done already

    :return: None
    """
    __check_prepared__(DEFAULT_EMBEDDING)
    app = application = bottle.default_app()
    bottle.run(server='gunicorn', host='127.0.0.1', port=PORT)


def __check_prepared__(_embedding):
    if not _embedding in ['word2vec', 'glove']:
        _embedding = DEFAULT_EMBEDDING

    if _embedding == 'word2vec':
        # Check if word2vec is loaded in RAM
        if word2vec_embeddings is None:
            __prepare__(_word2vec=True, _glove=False)

    if _embedding == 'glove':
        if glove_embeddings is None:
            __prepare__(_word2vec=False, _glove=True)


def __prepare__(_word2vec=True, _glove=False):
    """
        **Call this function prior to doing absolutely anything else.**

        :param None
        :return: None
    """
    global word2vec_embeddings, glove_embeddings

    if DEBUG: print("embeddings_interface: Loading Word Vector to Memory.")

    if _word2vec:
        word2vec_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            'resources/GoogleNews-vectors-negative300.bin', binary=True)

    if _glove:
        try:
            glove_embeddings = pickle.load(open(os.path.join(glove_location['dir'], glove_location['parsed'])))
        except IOError:
            # Glove is not parsed and stored. Do it.
            if DEBUG: warnings.warn(" GloVe is not parsed and stored. This will take some time.")

            glove_embeddings = {}
            f = open(os.path.join(glove_location['dir'], glove_location['raw']))
            iterable = f

            for line in iterable:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                glove_embeddings[word] = coefs
            f.close()

            # Now convert this to a numpy object
            pickle.dump(glove_embeddings, open(os.path.join(glove_location['dir'], glove_location['parsed']), 'w+'))

            if DEBUG: print("GloVe successfully parsed and stored. This won't happen again.")


def __congregate__(_vector_set, ignore=[]):
    if len(ignore) == 0:
        return np.mean(_vector_set, axis=0)
    else:
        return np.dot(np.transpose(_vector_set), ignore) / sum(ignore)


def __vectorize__(data):
    """
        Function to embed a sentence and return it as a list of vectors.
        WARNING: Give it already split. I ain't splitting it for ye.

        :param data: a dictionary containing some of the following:
                    _tokens: List of Strings
                    _encode_special_chars: Bool if special chars are to be encoded as well (for paths)
                    __embedding: String-  either 'word2vec' or 'glove'
        :return: Numpy tensor of n * 300d, [OPTIONAL] List(str) of tokens out of vocabulary.
    """
    # Parse args from the dict
    _tokens = data['_tokens']

    try: _encode_special_chars = data['_encode_special_chars']
    except KeyError: _encode_special_chars = False

    try:
        _embedding = data['_embedding']
        if _embedding not in ["glove", "word2vec"]: raise ValueError
    except KeyError: _embedding = DEFAULT_EMBEDDING
    except ValueError: _embedding = DEFAULT_EMBEDDING

    __check_prepared__(_embedding)

    op = []
    for token in _tokens:

        # Small cap everything
        token = token.lower()

        try:
            if _embedding == "glove":
                token_embedding = glove_embeddings[token]
            elif _embedding == 'word2vec':
                token_embedding = word2vec_embeddings.word_vec(token)

        except KeyError:
            token_embedding = np.zeros(300, dtype=np.float32)

        finally:

            if _encode_special_chars:
                # If you want path dividers like +, - or / to be treated specially
                if token == "+":
                    token_embedding = np.repeat(1, 300)
                elif token == "-":
                    token_embedding = np.repeat(-1, 300)
                elif token == "/":
                    token_embedding = np.repeat(0.5, 300)
            op += [token_embedding]

    return np.asarray(op)


@get('/vectorize')
def vectorize():
    """
        Function to embed a sentence and return it as a list of vectors.
    """
    try:
        # Gather input data
        try:
            data = request.json
        except:
            raise ValueError

        if data is None:
            raise ValueError

        # Check if the data is a dict, with all necessary things
        if type(data) == dict and \
                {'_embedding', '_encode_special_chars', '_report_unks', '_tokens'}.issuperset(set(data.keys())):
            pass
        else:
            raise TypeError

    except ValueError:
        # No data at all, raise error.
        response.status = 400
        return

    except TypeError:
        # Invalid data. Raise Error.
        response.status = 409
        return

    vectors = __vectorize__(data)

    # Serialize the vectors
    vectors = vectors.tolist()

    # Return 200 Success
    response.headers['Content-Type'] = 'application/json'
    return json.dumps({'result': vectors,
                       'request': data})


# def __phrase_similarity__(_phrase_1, _phrase_2, embedding='word2vec'):
def __phrase_similarity__(data):
    # Parse Arguments from data
    _phrase_1 = data["_phrase_1"]
    _phrase_2 = data["_phrase_2"]

    try:
        embedding = data["embedding"]
        if embedding not in ["glove", "word2vec"]: raise ValueError
    except KeyError: embedding = DEFAULT_EMBEDDING
    except ValueError: embedding = DEFAULT_EMBEDDING

    __check_prepared__(embedding)

    phrase_1 = _phrase_1.split(" ")
    phrase_2 = _phrase_2.split(" ")
    vw_phrase_1 = []
    vw_phrase_2 = []
    for phrase in phrase_1:
        try:
            # print phrase
            vw_phrase_1.append(word2vec_embeddings.word_vec(phrase.lower()) if embedding == 'word2vec'
                else glove_embeddings[phrase.lower()])
        except:
            # print traceback.print_exc()
            continue
    for phrase in phrase_2:
        try:
            vw_phrase_2.append(word2vec_embeddings.word_vec(phrase.lower()) if embedding == 'word2vec'
                else glove_embeddings[phrase.lower()])
        except:
            continue
    if len(vw_phrase_1) == 0 or len(vw_phrase_2) == 0:
        return 0
    v_phrase_1 = __congregate__(vw_phrase_1)
    v_phrase_2 = __congregate__(vw_phrase_2)
    cosine_similarity = np.dot(v_phrase_1, v_phrase_2) / (np.linalg.norm(v_phrase_1) * np.linalg.norm(v_phrase_2))
    return float(cosine_similarity)


@get('/phrase_similarity')
def phrase_similarity():
    """
            Function to compute cosine b/w two phrases
        """
    try:
        # Gather input data
        try:
            data = request.json
        except:
            raise ValueError

        if data is None:
            raise ValueError

        # Check if the data is a dict, with all necessary things
        if type(data) == dict and \
                {'_phrase_1', '_phrase_2', 'embedding'}.issuperset(set(data.keys())):
            pass
        else:
            raise TypeError

    except ValueError:
        # No data at all, raise error.
        response.status = 400
        return

    except TypeError:
        # Invalid data. Raise Error.
        response.status = 409
        print data
        return

    similarity_score = __phrase_similarity__(data)      # Serializable data

    # Return 200 Success
    response.headers['Content-Type'] = 'application/json'
    return json.dumps({'result': similarity_score,
                       'request': data})


if __name__ == '__main__':
    os.chdir('..')
    start()
