'''
    >File starts a simple bottle server which can then be used for accessing Krantikari.
        >it inits the model as well as all other peripherals needed

    >Takes three arguments
        1 - URL
        2 - PORT
        3 - GPU number
        Note - If GPU number is -1 then model is loaded on CPU instead of GPU

    >urls
        /answer -> returns answers to the user question
        /sparql -> returns sparql formed for the user question


    >template requests code

        import requests
        question = 'What is the capital of India ?'
        headers = {'Accept': 'text/plain', 'Content-type': 'application/json'}
        answer = requests.get('http://localhost:9000/answer',data={'question':question},headers=headers)
        print answer.content

    >possible error codes are
        'no_entity' --> No entity returned by the entity linker
        'no_best_path' --> No candidate paths created
        'entity_server_error' --> server issues at entity linking server
        '500' --> 'Internal Server Error'

    >Curl request
    curl -X 'GET' -H 'Accept: text/plain' -H 'Accept-Encoding: gzip, deflate' -H 'Connection: keep-alive' -H 'Content-Length: 41' -H 'Content-type: application/json' -H 'User-Agent: python-requests/2.18.4' -d 'question=What+is+the+capital+of+India+%3F' 'http://localhost:9000/answer'

    >To receive sparql, instead of sending the request at /answer send it on /sparql . Currently ask queries are not supported.
'''


import sys
import json
import traceback
from bottle import get, request, run, response, HTTPError




URL = 'localhost'
PORT = 9000
GPU = 3

def start():
    run(host=URL, port=PORT)


@get('/answer')
def find_node():
    try:
        # headers = {'Accept': 'text/plain', 'Content-type': 'application/json'}
        # b = requests.get('http://localhost:9000/answer',data={'question':a},headers=headers)
        question = request.forms['question']
        sparql, query_graph, error_code = dep.return_sparql(model_corechain, model_rdf_type_check, model_rdf_type_existence, model_question_intent,question)
        if error_code == '':
            answer = dep.retrive_answer(sparql,query_graph['intent'])
            response.content_type = 'application/json'
            return json.dumps(answer)
        else:
            return json.dumps(error_code)
    except:
        return HTTPError(500, 'Internal Server Error')


@get('/sparql')
def find_node():
    # headers = {'Accept': 'text/plain', 'Content-type': 'application/json'}
    # b = requests.get('http://localhost:9000/answer',data={'question':a},headers=headers)
    question = request.forms['question']
    sparql, query_graph, error_code = dep.return_sparql(question)
    if error_code == '':
        # answer = dep.retrive_answer(sparql,query_graph['intent'])
        response.content_type = 'application/json'
        if query_graph['intent'] == 'ask':
            return json.dumps('ask query not supported')
        return json.dumps(sparql)
    else:
        return json.dumps(error_code)



if __name__== "__main__":
    try:
        URL = sys.argv[1]
    except IndexError:
        pass
    try:
        PORT = int(sys.argv[2])
    except IndexError, TypeError:
        pass
    try:
        GPU = int(sys.argv[3])
    except:
        pass

    print "About to start server on %(url)s:%(port)s" % {'url':URL, 'port':str(PORT), 'gpu':str(GPU)}
    print "initilizing models and databases ... "

    import deploy as dep
    if GPU == -1:
        dep.DEVICE = 'CPU'
        dep.GPU = -1
    else:
        dep.DEVICE = 'GPU'
        dep.GPU = GPU

    model_corechain, model_rdf_type_check, model_rdf_type_existence, model_question_intent = dep.run(dep.DEVICE,dep.GPU)

    try:
        sparql, query_graph, error_code = dep.return_sparql(model_corechain, model_rdf_type_check, model_rdf_type_existence, model_question_intent,"what is the capital of India ?")
        print "done initilizing model"
    except:
        print "some issues initilizing model"
        print traceback.print_exc()
    start()