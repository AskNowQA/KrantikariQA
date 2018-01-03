"""
    Script does all the magic with word embeddings including lazy loading them in the RAM and all that.

    @TODO: Check how well is this performing
"""
import os
import json
import requests

# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def phrase_similarity(_phrase_1, _phrase_2, embedding='word2vec'):

    # Convert the params into

    phrase_1 = _phrase_1.split(" ")
    phrase_2 = _phrase_2.split(" ")
    vw_phrase_1 = []
    vw_phrase_2 = []
    for phrase in phrase_1:
        try:
            # print phrase
            vw_phrase_2.append(word2vec_embeddings.word_vec(phrase.lower() if embedding == 'word2vec'
                else glove_embeddings[phrase.lower()]))
        except:
            # print traceback.print_exc()
            continue
    for phrase in phrase_2:
        try:
            vw_phrase_2.append(word2vec_embeddings.word_vec(phrase.lower() if embedding == 'word2vec'
                else glove_embeddings[phrase.lower()]))
        except:
            continue
    if len(vw_phrase_1) == 0 or len(vw_phrase_2) == 0:
        return 0
    v_phrase_1 = __congregate__(vw_phrase_1)
    v_phrase_2 = __congregate__(vw_phrase_2)
    cosine_similarity = np.dot(v_phrase_1, v_phrase_2) / (np.linalg.norm(v_phrase_1) * np.linalg.norm(v_phrase_2))
    return float(cosine_similarity)


def vectorize():
    """

    :return:
    """

    # try:
    #     # parse input data
    #     try:
    #         data = request.json()
    #     except:
    #         raise ValueError
    #
    #     if data is None:
    #         raise ValueError
    #
    # except ValueError:
    #     # if bad request data, return 400 Bad Request
    #     response.status = 400
    #     return
    #
    # except KeyError:
    #     # if name already exists, return 409 Conflict
    #     response.status = 409
    #     return
    #
    # # add nam

    # return 200 Success
    response.headers['Content-Type'] = 'application/json'
    return json.dumps({'name': 'name'})
