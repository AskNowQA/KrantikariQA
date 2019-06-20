'''

    This file interacts with other two servers (simpleQA server as well as Krantikari QA server.) It first hits the hop server which decides wheter
    the question is a single hop or a two hop question.
'''
import sys
import requests, os, json
from bottle import get, request, run, response, HTTPError


os.environ['NO_PROXY'] = 'localhost'

HOP_SERVER_URL = 'http://localhost:6969/'
SIMPLE_QA_URl = 'http://localhost:9000/'
KRANTIKARI_QA_URL = 'http://localhost:10000/'

DEBUG = False


@get('/graph')
def answer():
    try:
        question = request.forms['question']

        if DEBUG:
            print(f"recived question at meta server is {question}")
        # Send this to hop server.

        headers = {'Accept': 'text/plain', 'Content-type': 'application/json'}
        answer = requests.get(HOP_SERVER_URL + 'hop', data={'question': question}, headers=headers)
        hops = json.loads(answer.content.decode('utf-8'))['hops']

        if DEBUG:
            print(f"number of hops predicted is {hops}")

        if hops == 1:
            # hit the first server
            return requests.get(SIMPLE_QA_URl + 'graph', data={'question': question}, headers=headers)
        else:
            return requests.get(KRANTIKARI_QA_URL + 'graph', data={'question': question}, headers=headers)

    except:
        raise HTTPError(500, 'Internal Server Error')

@get('/answer')
def answer_with_entites():
    try:
        question = request.forms['question']
        entites = requests.forms['requests']
        # Send this to hop server.

        if DEBUG:
            print(f"recived question and entity at meta server is respectively is {question}, {entites}")

        headers = {'Accept': 'text/plain', 'Content-type': 'application/json'}
        answer = requests.get(HOP_SERVER_URL + 'hop', data={'question': question}, headers=headers)
        hops = json.loads(answer.content.decode('utf-8'))['hops']

        if DEBUG:
            print(f"number of hops predicted is {hops}")

        if hops == 1:
            # hit the first server
            return requests.get(SIMPLE_QA_URl,
                                         data={'question':question,
                                               'entities':json.dumps(entites)},headers=headers)
        else:
            return requests.get(KRANTIKARI_QA_URL,
                                data={'question': question,
                                      'entities': json.dumps(entites)}, headers=headers)
    except:
        raise HTTPError(500, 'Internal Server Error')

if __name__ == "__main__":
    try:
        URL = sys.argv[1]
    except IndexError:
        pass
    try:
        PORT = int(sys.argv[2])
    except (IndexError, TypeError):
        pass

    GPU = 'not supported for now.'

    print("About to start server on %(url)s:%(port)s" % {'url': URL, 'port': str(PORT), 'gpu': str(GPU)})
    print("initilizing models and databases ... ")

    run(host=URL, port=PORT)
