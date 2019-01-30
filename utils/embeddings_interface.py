import os
import sys
import torch
import pickle
import warnings
import traceback
import numpy as np
import progressbar #progressbar2

# This code will NOT work locally.
sys.path.append('/data/priyansh/conda/fastai')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from fastai.text import *

DEBUG = True
vectors, vocab, temp_vocab_id = [], {}, {'_pad_':(0,False), '_unk_':(1,False), '+':(2,False), '-':(3,False),
                                         '/':(4,False), 'uri':(5,False), 'x':(6,False)}

POSSIBLE_EMBEDDINGS = ['glove', 'fasttext', 'ulmfit']
DEFAULT_EMBEDDING = POSSIBLE_EMBEDDINGS[0]
SELECTED_EMBEDDING = None

SPECIAL_CHARACTERS = ['_pad_', '_unk_', '+', '-', '/', 'uri', 'x']
SPECIAL_EMBEDDINGS = [0, 0, 1, -1, 0.5, -2, 2]
GLOVE_LENGTH = 2196017

EMBEDDING_DIM = 400
EMBEDDING_GLOVE_DIM = 300
EMBEDDING_FASTAI_DIM = 300  # @TODO: fix

PREPARED = False

parsed_location = './resources'

glove_location = \
    {
        'dir': "./resources",
        'raw': "glove.840B.300d.txt",
        'vec': "vectors_gl.npy",
        'voc': "vocab_gl.pickle"
    }
fasttext_location = \
    {
        'dir': "./resources",
        'raw': "wiki-news-300d-1M.vec",
        'vec': "vectors_fa.npy",
        'voc': "vocab_fa.pickle"
    }
ulmfit_location = \
    {
        'dir': "./resources/ulmfit/wt103",
        'raw_voc': "itos_wt103.pkl",
        'raw_vec': "fwd_wt103_enc.h5",
        'vec': "vectors_ul.npy",
        'voc': "vocab_ul.pickle"
    }


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


class NotYetImplementedError(Exception):
    pass


def __check_prepared__(_embedding=None):
    if len(vectors) <= len(SPECIAL_CHARACTERS) or \
            len(vocab) <= len(SPECIAL_CHARACTERS) or \
            (_embedding != None and _embedding != SELECTED_EMBEDDING):
        __prepare__(_embedding)


def __prepare__(_embedding=None):
    global SELECTED_EMBEDDING, EMBEDDING_DIM

    # If someone gave an embedding, mark that as the permanent one
    SELECTED_EMBEDDING = _embedding if _embedding != None else DEFAULT_EMBEDDING
    EMBEDDING_DIM = 300 if SELECTED_EMBEDDING in ['glove'] else 400

    _init_special_characters_()

    if SELECTED_EMBEDDING == 'ulmfit':
        _parse_ulmfit_()

    if SELECTED_EMBEDDING == 'glove':
        _parse_glove_()

    if SELECTED_EMBEDDING == 'fasttext':
        warnings.warn("Haven't implemented Fasttext parser yet.")
        raise NotYetImplementedError


def _init_special_characters_():
    """
        Regardless of whatever we choose, vectors and vocab need to have basic stuff in them.
        Depends on what we mention as special characters.

        This fn assumes empty vector, vocab
    """
    global vectors, vocab

    try:
        assert len(vectors) == 0 & len(vocab) == 0
    except AssertionError:
        warnings.warn("Found non empty vectors, vocab. Cleaning them up.")
        for sp_char in SPECIAL_CHARACTERS:
            assert sp_char in vocab


    # Push special chars in the vocab, alongwith their vectors IF not already there.
    for i, sp_char in enumerate(SPECIAL_CHARACTERS):
        vocab[sp_char] = i
        vectors.append(np.repeat(SPECIAL_EMBEDDINGS[i], EMBEDDING_DIM))


def __parse_line__(line):
    """
        Used for glove raw file parsing.

        Partitions the list into two depending on till where in it do words exist.
        e.g. the 1 2 3 will be 'the' [1 2 3]
        eg. the person 1 2 will be 'the person' [1 2]
    """
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    word = [tokens[0]]
    tokens = tokens[1:]
    while True:
        token = tokens[0]
        #         print(tokreen)
        try:
            _ = float(token)
            break
        except ValueError:
            word.append(token)
            tokens.pop(0)

    #     print(line)
    assert len(tokens) == 300  # Hardcoded here, because we know glove pretrained is 300d
    #     raise EOFError
    return ' '.join(word), np.asarray(tokens)


def _parse_glove_():
    """
        Fn to go through glove's raw file, and add vocab, vectors for words not already in vocab.
    """
    global vectors, vocab

    print("Loading Glove vocab and vectors from disk. Sit Tight.")

    try:
        # Try to load from disk
        vocab, vectors = load(_embedding='glove')

        return True

    except FileNotFoundError:

        warnings.warn("Couldn't find Glove vocab and/or vectors on disk. Parsing from raw file will TAKE TIME ...")

        # Assume that vectors can be list OR numpy array.

        changes = 0
        lines = 0
        new_vectors = []

        # Open raw file
        f = open(os.path.join(glove_location['dir'], glove_location['raw']))

        if DEBUG:
            max_value = progressbar.UnknownLength if GLOVE_LENGTH is None else GLOVE_LENGTH
            bar = progressbar.ProgressBar(max_value=max_value)

        for line in f:
            lines += 1
            # Parse line
            word, coefs = __parse_line__(line)

            # Ignore if word is a special char
            if word in SPECIAL_CHARACTERS:
                continue

            # Ignore if we already have this word
            try:
                _ = vocab[word]
                continue
            except KeyError:
                # Its a new word, put it in somewhere.
                vocab[word] = len(vocab)
                new_vectors.append(coefs)
                changes += 1

            if DEBUG:
                bar.update(lines)

        f.close()

        # Merge vectors
        new_vectors = np.array(new_vectors)
        vectors = np.array(vectors)

        if DEBUG:
            print("Old vectors: ", vectors.shape)
            print("New vectors: ", new_vectors.shape)

        vectors = np.vstack((vectors, new_vectors))

        if DEBUG:
            print("Combined vectors: ", vectors.shape)
            print("Vocab: ", len(vocab))

        save()

        return True


def _parse_ulmfit_():
    global vocab, vectors

    print("Loading ULMFIT vocab and vectors from disk. Sit Tight.")

    try:
        # Try to load from disk
        vocab, vectors = load(_embedding='ulmfit')

        return True

    except FileNotFoundError:

        warnings.warn("Couldn't find ULMFIT vocab and/or vectors on disk. Parsing them from raw. Any second now ...")

        # Load ulmfit vectors and vocab in mem
        ulmfit_words = pickle.load(open(
            os.path.join(ulmfit_location['dir'], ulmfit_location['raw_voc']), 'rb'))
        ulmfit_vocab = {word: index for index, word in enumerate(ulmfit_words)}
        ulmfit_model = torch.load(os.path.join(ulmfit_location['dir'], ulmfit_location['raw_vec']),
                                  map_location=lambda storage, loc: storage)
        ulmfit_vectors = to_np(ulmfit_model['encoder.weight'])

        to_add_char = []
        for sp_char in SPECIAL_CHARACTERS:
            try:
                sp_char_id = ulmfit_vocab[sp_char]
            except KeyError:
                ulmfit_vocab[sp_char_id] = len(ulmfit_vocab)
                to_add_char.append(sp_char_id)

        if DEBUG:
            print(to_add_char)
            print("Vocab :", len(ulmfit_vocab))
            print("Vectors: ", ulmfit_vectors.shape)

        if len(to_add_char) > 0:
            # Add these things to vectors
            new_vectors = []
            for char in to_add_char:
                newid = SPECIAL_CHARACTERS.index(char)
                vec = np.repeat(SPECIAL_EMBEDDINGS[newid], EMBEDDING_DIM)
                new_vectors.append(vec)

            new_vectors = np.asarray(new_vectors)
            print(new_vectors.shape, ulmfit_vectors.shape)
            vectors = np.vstack((ulmfit_vectors, new_vectors))
        else:
            vectors = ulmfit_vectors

        vocab = ulmfit_vocab

        # Replace pad and unk tokens
        vocab['_unk_'] = 1
        vocab['_pad_'] = 0
        vectors[[0, 1]] = vectors[[1, 0]]

        if DEBUG:
            print("Vocab :", len(vocab))
            print("Vectors: ", vectors.shape)

        save()

        return True


def save():
    if SELECTED_EMBEDDING == 'glove':
        locs = glove_location
    elif SELECTED_EMBEDDING == 'ulmfit':
        locs = ulmfit_location
    elif SELECTED_EMBEDDING == 'fasttext':
        raise NotYetImplementedError

    if DEBUG:
        print("Saving %(emb)s in %(loc)s" % {'emb': SELECTED_EMBEDDING, 'loc': parsed_location})

    # Save vectors
    np.save(os.path.join(parsed_location, locs['vec']), vectors)

    # Save vocab
    pickle.dump(vocab, open(os.path.join(parsed_location, locs['voc']), 'wb+'))


def load(_embedding):
    local_vocab, local_vectors = {}, []

    if _embedding == 'glove':
        locs = glove_location
    elif _embedding == 'ulmfit':
        locs = ulmfit_location
    elif _embedding == 'fasttext':
        raise NotYetImplementedError

    local_vocab = pickle.load(open(os.path.join(parsed_location, locs['voc']), 'rb'))
    local_vectors = np.load(os.path.join(parsed_location, locs['vec']))

    return local_vocab, local_vectors


def vocabularize(_tokens, _report_unks=False, _case_sensitive=False, _embedding=None):
    """
        Given a list of strings (list of tokens), this returns a list of integers (vectorspace ids)
        based on whatever embedding is called in the function, or is the SELECTED EMBEDDING

        :param _tokens: The sentence you want vocabbed. Tokenized (list of str)
        :param _report_unks: Whether or not to return the list of out of vocab tokens
        :param _case_sensitive: Whether or not return to lowercase everything.
        :param _embedding: Which embeddings do you want to use in this process.
        :return: Numpy tensor of n, [OPTIONAL] List(str) of tokens out of vocabulary.
    """
    __check_prepared__(_embedding=_embedding)

    op = []
    unks = []

    for token in _tokens:

        # Small cap everything
        token = token.lower() if not _case_sensitive else token

        try:

            try:

                token_id = vocab[token]

            except KeyError:
                '''
                    It doesn't exist, give it the _unk_ token, add log it to unks.
                '''
                if _report_unks: unks.append(token)
                token_id = vocab['_unk_']

            finally:

                op += [token_id]
        except:

            "This here is to prevent some unknown mishaps, which stops a real long process, and sends me hate mail."
            print(traceback.print_exc())
            print(token)

    return (np.asarray(op), unks) if _report_unks else np.asarray(op)


def vocabularize_idspace(_tokens, _unk=True, _case_sensitive=False, _embedding=None):
    """
        Given a list of strings (list of tokens), this returns a list of integers (vectorspace ids)
        based on whatever embedding is called in the function, or is the SELECTED EMBEDDING

        :param _tokens: The sentence you want vocabbed. Tokenized (list of str)
        :param _vocab: A dictionary containing word and an id of that word.
        :param _report_unks: Whether or not to return the list of out of vocab tokens
        :param _case_sensitive: Whether or not return to lowercase everything.
        :param _embedding: Which embeddings do you want to use in this process.
        :return: Numpy tensor of n, [OPTIONAL] List(str) of tokens out of vocabulary.
    """
    # __check_prepared__(_embedding=_embedding)

    global temp_vocab_id
    op = []


    for token in _tokens:

        # Small cap everything
        token = token.lower() if not _case_sensitive else token

        try:

            try:

                token_id = temp_vocab_id[token][0]

            except KeyError:
                '''
                    It doesn't exist, give it the _unk_ token, add log it to unks.
                '''
                token_id = len(temp_vocab_id)
                temp_vocab_id[token] = (token_id,_unk)

            finally:

                op += [token_id]
        except:

            "This here is to prevent some unknown mishaps, which stops a real long process, and sends me hate mail."
            print(traceback.print_exc())
            print(token)

    return np.asarray(op)

def vectorize(_tokens, _report_unks=False, _case_sensitive=False, _embedding=None):
    """
        Function to embed a sentence and return it as a list of vectors.
        WARNING: Give it already split. I ain't splitting it for ye.

        :param _tokens: The sentence you want embedded. (Assumed pre-tokenized input)
        :param _report_unks: Whether or not return the out of vocab words
        :param _case_sensitive: Whether or not return to lowercase everything.
        :param _embedding: Which embeddings do you want to use in this process.
        :return: Numpy tensor of n * 300d, [OPTIONAL] List(str) of tokens out of vocabulary.
    """

    __check_prepared__(_embedding)

    op = []
    unks = []
    for token in _tokens:

        # Small cap everything
        token = token.lower() if not _case_sensitive else token

        try:

            token_embedding = vectors[vocab[token]]

        except KeyError:

            if _report_unks: unks.append(token)
            token_embedding = vectors[vocab['_unk_']]

        finally:

            op += [token_embedding]

    return (np.asarray(op), unks) if _report_unks else np.asarray(op)


def __congregate__(_vector_set, ignore=[]):
    if len(ignore) == 0:
        return np.mean(_vector_set, axis=0)
    else:
        return np.dot(np.transpose(_vector_set), ignore) / sum(ignore)


def phrase_similarity(_phrase_1, _phrase_2, embedding='glove'):
    """
        Legacy Function. Don't know where and why is it used. :/
    """

    __check_prepared__(embedding)

    phrase_1 = _phrase_1.split(" ")
    phrase_2 = _phrase_2.split(" ")
    vw_phrase_1 = []
    vw_phrase_2 = []
    for phrase in phrase_1:
        try:
            # print phrase
            vw_phrase_1.append(vectors[vocab[phrase.lower()]])
        except:
            # print traceback.print_exc()
            continue
    for phrase in phrase_2:
        try:
            vw_phrase_2.append(vectors[vocab[phrase.lower()]])
        except:
            continue
    if len(vw_phrase_1) == 0 or len(vw_phrase_2) == 0:
        return 0
    v_phrase_1 = __congregate__(vw_phrase_1)
    v_phrase_2 = __congregate__(vw_phrase_2)
    cosine_similarity = np.dot(v_phrase_1, v_phrase_2) / (np.linalg.norm(v_phrase_1) * np.linalg.norm(v_phrase_2))
    return float(cosine_similarity)


def update_vocab(_words, _embedding=None):
    """
        Function to add new words to an existing vocab and save it to disk.
        words = ['potato', 'gauravmaheshwari', 'rabbithole', 'bumhole', 'adnsakndksnadnsad']

    :param _words: list of str (see above)
    :return: None
    """


    global vocab, vectors
    __check_prepared__(_embedding=_embedding)


    old_len = len(vocab)
    new_vocab = {word: vocab.get(word, len(vocab) + i) for i, word in enumerate(_words)}
    vocab.update(new_vocab)
    new_len = len(vocab)


    # Need new vectors for all these new words.
    new_vectors = np.random.randn(new_len - old_len, EMBEDDING_DIM)
    print("shape of new_vectors is ", new_vectors.shape)
    print("shape of vectors is ", vectors.shape)
    vectors = np.vstack((vectors, new_vectors))

    save()


def align_id_space():
    global temp_vocab_id, vocab, vectors

    new_vectors = []
    new_vocab = {}

    rev = {v[0]:k for k,v in temp_vocab_id.items()}


    for id in range(len(rev)):
        token = rev[id]
        value = temp_vocab_id[token]

        new_vocab[token] = value[0]

        try:

            vec = vectors[vocab[token]]

        except KeyError:
            '''
                It doesn't exist, give it the _unk_ token, add log it to unks.
            '''
            if value[1]:
                vec = vectors[vocab['_unk_']]
            else:
                vec = np.random.randn(EMBEDDING_DIM)

        finally:

            new_vectors.append(vec)

    #converting to numpy
    new_vectors = np.asarray(new_vectors)

    #save them somewhere
    if SELECTED_EMBEDDING == 'glove':
        locs = glove_location
    elif SELECTED_EMBEDDING == 'ulmfit':
        locs = ulmfit_location
    elif SELECTED_EMBEDDING == 'fasttext':
        raise NotYetImplementedError

    if DEBUG:
        print("Saving %(emb)s in %(loc)s" % {'emb': SELECTED_EMBEDDING, 'loc': parsed_location})

    # Save vectors
    np.save(os.path.join(parsed_location, locs['vec']), new_vectors)

    # Save vocab
    pickle.dump(new_vocab, open(os.path.join(parsed_location, locs['voc']), 'wb+'))


