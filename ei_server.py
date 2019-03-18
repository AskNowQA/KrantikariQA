from bottle import get, request, run, response, HTTPError
from utils import embeddings_interface
from utils import natural_language_utilities as nlutils
import numpy as np

import json
@get('/vec')
def vocabularize():
    question = request.json['question']
    vec = embeddings_interface.vectorize(nlutils.tokenize(question), _embedding=EMBEDDING).astype(np.float).tolist()
    return json.dumps(vec)

if __name__ == '__main__':

    embeddings_interface.__check_prepared__()

    EMBEDDING = 'glove'
    URL = 'localhost'
    PORT = 3500

    print("About to start server on %(url)s:%(port)s" % {'url': URL, 'port': str(PORT)})
    print("initilizing models and databases ... ")

    run(host=URL, port=PORT,workers=2)