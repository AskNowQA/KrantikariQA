"""
    Script does all the magic with word embeddings including lazy loading them in the RAM and all that.

    ! NOTE: This file no longer works with word2vec. Pls stay tuned.

    For handeling out of vocab run the embedding interface such that it sees the whole data set i.e. run krantikari step 2. Once that is done, delete the glove_vocab.pickle
    and glove_parssed.npy. Then run the krantikari_step_2 again. This will include out of vocab for the data seen by krantikari.py. One can do exactly same for relation aggregate.
     Summarize - Whenever a new data is seen, this three step process of
	>Run the file
	>Delete vocab
	>Run the file again.
	needs to be done.

"""

import os
import json
import gensim
import pickle
import bottle
import warnings
import traceback
import numpy as np

from bottle import post, get, put, delete, request, response
# from torch.nn._functions.thnn.pooling import MaxPool2d

word2vec_embeddings = None
glove_embeddings = None
glove_vocab = None
DEFAULT_EMBEDDING = 'word2vec'
DEBUG = True
oov_counter = 0
glove_location = \
    {
        'dir': "./resources",
        'raw': "glove.42B.300d.txt",
        'parsed': "glove_parsed.npy",
        'vocab': "glove_vocab.pickle"
    }

OUT_OF_VOCAB = 'resources/out_of_vocab.pickle'
MAX_GLOVE_LENGTH = 0
out_of_vocab = []   #keeps track of all the out of vocab words

# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def __check_prepared__(_embedding, _only_vocab=False):
    if not _embedding in ['word2vec', 'glove']:
        _embedding = DEFAULT_EMBEDDING

    if _embedding == 'word2vec':
        # Check if word2vec is loaded in RAM
        if word2vec_embeddings is None:
            __prepare__(_word2vec=True, _glove=False, _only_vocab=_only_vocab)

    if _embedding == 'glove':
        if glove_vocab is None:
            __prepare__(_word2vec=False, _glove=True, _only_vocab=_only_vocab)
        elif _only_vocab is False and glove_embeddings is None:
            __prepare__(_word2vec=False, _glove=True, _only_vocab=_only_vocab)


def __prepare__(_word2vec=True, _glove=False, _only_vocab=False):
    """
        **Call this function prior to doing absolutely anything else.**

        :param None
        :return: None
    """
    global word2vec_embeddings, glove_embeddings, glove_vocab

    if DEBUG: print("embeddings_interface: Loading Word Vector to Memory.")

    if _word2vec:
        word2vec_embeddings = gensim.models.KeyedVectors.load_word2vec_format('resources/GoogleNews-vectors-negative300.bin', binary=True)

    if _glove:

        """
            Two step process:
                - read the file for vocabs,
                - if needed (param), read the file again for corresponding embeddings.
        """

        # Forgo loading vocab if it is already in memory.

        if glove_vocab is None:

            try:
                glove_vocab = pickle.load(open(os.path.join(glove_location['dir'], glove_location['vocab'])))
                '''
                    Load the OOV word with their id's
                '''
                counter = len(glove_vocab) + 1
                if DEBUG: print "total vocab size is ", str(counter-1)
                try:
                    #check if oov file exists
                    oov = pickle.load(open(OUT_OF_VOCAB))
                    for token in oov:
                        glove_vocab[token] = counter
                        counter = counter + 1
                    if DEBUG: print "length of new vocab file is ", len(glove_vocab)
                except:
                    print traceback.print_exc()
                    if DEBUG: print "oov file not found @glove_vocab"
                    pass
            except (IOError, EOFError) as e:
                if DEBUG: warnings.warn(" GloVe vocabulary is not parsed and stored. This will take some time.")

                glove_vocab = {}

                # Push Special chars artificially.
                glove_vocab['PAD'] = 0
                glove_vocab['UNK'] = 1
                glove_vocab['+'] = 2
                glove_vocab['-'] = 3
                glove_vocab['/'] = 4
                glove_vocab['uri'] = 5
                glove_vocab['x'] = 6

                f = open(os.path.join(glove_location['dir'], glove_location['raw']))
                counter = 7
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in ['PAD', 'UNK', '+', '-', '/','uri','x']:
                        continue
                    glove_vocab[word] = counter
                    counter += 1
                MAX_GLOVE_LENGTH = counter + 1
                f.close()

                '''
                    Check for vocab file.
                '''

                try:
                    vocab_list = pickle.load(open(OUT_OF_VOCAB))
                    '''
                        Remove all the oov tokens which are part of the
                    '''
                    for word in vocab_list:
                        glove_vocab[word] = MAX_GLOVE_LENGTH
                        MAX_GLOVE_LENGTH = MAX_GLOVE_LENGTH + 1
                    if DEBUG:
                        print "vocab file now part of vocab list"
                except:
                    pass
                # Now store this object
                pickle.dump(glove_vocab, open(os.path.join(glove_location['dir'], glove_location['vocab']), 'w+'))

                if DEBUG: print("GloVe vocab successfully parsed and stored. This won't happen again.")

        if _only_vocab: return None

        # Convert embeddings to numpy object
        # glove_embeddings = np.asarray(glove_embeddings)

        try:

            # Let's try to load the embeddings now.
            glove_embeddings = np.load(open(os.path.join(glove_location['dir'], glove_location['parsed'])))
            #Now load the oov words
            try:
                oov = pickle.load(open(OUT_OF_VOCAB))
                for token in oov:
                    glove_embeddings[glove_vocab[token]] = oov[token]
            except:
                if DEBUG: print('No oov file found during embedding interface loading')
                pass

        except IOError:
            # Glove is not parsed and stored. Do it.
            if DEBUG: warnings.warn(" GloVe embeddings are not parsed and stored. This will take some time.")

            glove_embeddings = np.zeros((len(glove_vocab.keys()), 300))
            f = open(os.path.join(glove_location['dir'], glove_location['raw']))

            try:
                oov = pickle.load(open(OUT_OF_VOCAB))
                for token in oov:
                    glove_embeddings[glove_vocab[token]] = oov[token]
            except:
                pass

            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                try:
                    glove_embeddings[glove_vocab[word]] = coefs
                except:
                    try:
                        if oov:
                            glove_embeddings[glove_vocab[word]] = oov[word]
                        else:
                            glove_embeddings[glove_vocab[word]] = np.random.rand(1,300)
                    except:
                        glove_embeddings[glove_vocab[word]] = np.random.rand(1, 300)
            f.close()

            # Now store them to disk
            np.save(os.path.join(glove_location['dir'], glove_location['parsed']), glove_embeddings)

            if DEBUG: print("GloVe embeddings successfully parsed and stored. This won't happen again.")


def __congregate__(_vector_set, ignore=[]):
    if len(ignore) == 0:
        return np.mean(_vector_set, axis = 0)
    else:
        return np.dot(np.transpose(_vector_set), ignore) / sum(ignore)


def phrase_similarity(_phrase_1, _phrase_2, embedding='glove'):

    __check_prepared__(embedding)

    phrase_1 = _phrase_1.split(" ")
    phrase_2 = _phrase_2.split(" ")
    vw_phrase_1 = []
    vw_phrase_2 = []
    for phrase in phrase_1:
        try:
            # print phrase
            vw_phrase_1.append(word2vec_embeddings.word_vec(phrase.lower()) if embedding == 'word2vec'
                else glove_embeddings[glove_vocab[phrase.lower()]])
        except:
            # print traceback.print_exc()
            continue
    for phrase in phrase_2:
        try:
            vw_phrase_2.append(word2vec_embeddings.word_vec(phrase.lower()) if embedding == 'word2vec'
                else glove_embeddings[glove_vocab[phrase.lower()]])
        except:
            continue
    if len(vw_phrase_1) == 0 or len(vw_phrase_2) == 0:
        return 0
    v_phrase_1 = __congregate__(vw_phrase_1)
    v_phrase_2 = __congregate__(vw_phrase_2)
    cosine_similarity = np.dot(v_phrase_1, v_phrase_2) / (np.linalg.norm(v_phrase_1) * np.linalg.norm(v_phrase_2))
    return float(cosine_similarity)


def vectorize(_tokens, _report_unks=False, _encode_special_chars=False, _embedding='glove'):
    """
        Function to embed a sentence and return it as a list of vectors.
        WARNING: Give it already split. I ain't splitting it for ye.

        :param _tokens: The sentence you want embedded. (Assumed pre-tokenized input)
        :param _report_unks: Whether or not return the out of vocab words
        :return: Numpy tensor of n * 300d, [OPTIONAL] List(str) of tokens out of vocabulary.
    """

    __check_prepared__(_embedding)

    op = []
    unks = []
    for token in _tokens:

        # Small cap everything
        token = token.lower()

        try:
            if _embedding == "glove":
                token_embedding = glove_embeddings[glove_vocab[token]]
            elif _embedding == 'word2vec':
                token_embedding = word2vec_embeddings.word_vec(token)

        except KeyError:
            if _report_unks: unks.append(token)
            token_embedding = np.repeat(-0.5, 300)

        finally:

            if _encode_special_chars:
                # If you want path dividers like +, - or / to be treated specially
                if token == "+":
                    token_embedding = np.repeat(1, 300)
                elif token == "-":
                    token_embedding = np.repeat(-1, 300)
                elif token == "/":
                    token_embedding = np.repeat(0.5, 300)
                elif token == "PAD":
                    token_embedding = np.zeros(300, dtype=np.float32)
                elif token == "x":
                    token_embedding = np.repeat(2, 300)
                elif token == "uri":
                    token_embedding = np.repeat(-2, 300)

            op += [token_embedding]

    return (np.asarray(op), unks) if _report_unks else np.asarray(op)


def vocabularize(_tokens, _report_unks=False, _embedding='glove'):
    """
            Function to embed a sentence and return it as a list of "IDS".
            WARNING: Give it already split. I ain't splitting it for ye.
            :param _tokens: The sentence you want embedded. (Assumed pre-tokenized input)
            :param _report_unks: Whether or not return the out of vocab words
            :return: Numpy tensor of n * 300d, [OPTIONAL] List(str) of tokens out of vocabulary.
        """

    __check_prepared__(_embedding, _only_vocab=True)

    op = []
    unks = []
    for token in _tokens:

        # Small cap everything
        token = token.lower()
        try:

            try:

                if _embedding == "glove":
                    token_id = glove_vocab[token]
                else:
                    token_id = glove_vocab[token]

            except KeyError:
                '''
                    Add it to out of vocab dictionary.
                    Init it with random 300D vector.
                '''
                # print "check"
                if token not in out_of_vocab:
                    out_of_vocab.append(token)
                    token_id = 0
                else:
                    if _report_unks:
                        unks.append(token)
                    global oov_counter
                    oov_counter = oov_counter + 1
                    token_id = 0
            finally:

                op += [token_id]
        except:
            print traceback.print_exc()
            print token
            print token_id

    return (np.asarray(op), unks) if _report_unks else np.asarray(op)

def save_out_of_vocab():
    global oov_counter
    print "out of vocab words are ", str(oov_counter)
    try:
        ov = pickle.load(open(OUT_OF_VOCAB))
        for token in  out_of_vocab:
            if token not in ov:
                ov[token] = np.random.rand(1,300)
        pickle.dump(ov,open(OUT_OF_VOCAB,'w+'))
    except:
        ov = {}
        for token in out_of_vocab:
            ov[token] = np.random.rand(1,300)
        pickle.dump(ov, open(OUT_OF_VOCAB, 'w+'))


