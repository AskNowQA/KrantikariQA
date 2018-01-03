"""
    Script does all the magic with word embeddings including lazy loading them in the RAM and all that.

    NOTE: Embedding parameter is ignored. Right off the bat. (Not removed for compatibility reasons)
"""
import os
import json
import warnings
import requests
import numpy as np
from utils import embedding_server

URL = "http://127.0.0.1:6969/"
FALLBACK_SIMILARITY_VAL = 0.0
FALLBACK_EMBEDDING_VAL = np.zeros(300, dtype=np.float32)
DEBUG = True


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def phrase_similarity(_phrase_1, _phrase_2, embedding='word2vec'):

    url = URL + 'phrase_similarity'

    # Convert the params into a dict
    data = {'_phrase_1': _phrase_1,
            '_phrase_2': _phrase_2}

    try:

        output = requests.get(url=url, json=data)

        # Ensure the server shat out the right stuff
        if not output.status_code == 200:
            raise ValueError

        # Parse the result.
        result = json.loads(output.text)

        # Return stuff
        return result["result"]

    except requests.ConnectionError:

        # The server is down. Start it up.
        warnings.warn("The server is down. Start it up.")
        return FALLBACK_SIMILARITY_VAL

    except ValueError:

        # Unexpected Status Code from the server.
        warnings.warn("Invalid Status Code. Returning default value for phrase: \n\t %(p1)s \n\t %(p2)s"
                      % {'p1': _phrase_1, 'p2': _phrase_2})
        return FALLBACK_SIMILARITY_VAL

    except KeyError:

        # This should not happen. We should not be here.
        warnings.warn("Cannot find \"result\" in the server output. This should not have happened. " +
                      "Scream and run around in circles")
        return FALLBACK_SIMILARITY_VAL


def vectorize():
    """

    :return:
    """
    pass
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
    # response.headers['Content-Type'] = 'application/json'
    # return json.dumps({'name': 'name'})


if __name__ == "__main__":
    print phrase_similarity("Obama", "Potato")