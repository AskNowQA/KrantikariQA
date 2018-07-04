# KrantikariQA
An Information Gain based Question Answering system over knowledge graph systems.


Note: Using stopwords from https://github.com/igorbrigadir/stopwords (found in resources/atire_puurula.txt)

### Running the server file.
server.py file starts a simple bottle server which can then be used for accessing Krantikari and also initializes all models and peripheral files needed.
```
python server.py localhost 9000 3
    >Takes three arguments
        1 - URL
        2 - PORT
        3 - GPU number
```

#### Template requests code

        import requests
        question = 'What is the capital of India ?'
        headers = {'Accept': 'text/plain', 'Content-type': 'application/json'}
        answer = requests.get('http://localhost:9000/answer',data={'question':question},headers=headers)
        print answer.content

#### Possible error codes
        'no_entity' --> No entity returned by the entity linker
        'no_best_path' --> No candidate paths created
        'entity_server_error' --> server issues at entity linking server
        '500' --> 'Internal Server Error'

#### Curl request
    curl -X 'GET' -H 'Accept: text/plain' -H 'Accept-Encoding: gzip, deflate' -H 'Connection: keep-alive' -H 'Content-Length: 41' -H 'Content-type: application/json' -H 'User-Agent: python-requests/2.18.4' -d 'question=What+is+the+capital+of+India+%3F' 'http://localhost:9000/answer'

#### Urls
        /answer -> returns answers to the user question
        /sparql -> returns sparql formed for the user question

To receive sparql, instead of sending the request at /answer send it on /sparql . Currently ask queries are not supported.
