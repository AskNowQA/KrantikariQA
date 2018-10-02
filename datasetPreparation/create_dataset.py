'''
    >Read node information
    >Send each SPARQL to the SPARQL parsing service
    >Send each response of the sparql parsing service to retrieve constraints, true paths and entites
    >Send the entity to the entity_subgraph class to retrive the required subgraph
    >Pack it in specific manner
'''
import requests
import qald_parser as qp
from datasetPreparation import entity_subgraph as es
from utils import dbpedia_interface as db_interface


class create_data_node():
    def __init__(self,_predicate_blacklist,_relation_file,_qald=False):
        self.dbp = db_interface.DBPedia(caching=False)
        self.relation_file = _relation_file
        self.predicate_blacklist = _predicate_blacklist
        self.qald = _qald
        self.create_subgraph = es.create_subgraph(self.dbp,self.predicate_blacklist,self.relation_file, qald=_qald)

    def parse_sparql(self,sparql_query):
        '''

        	Given a sparql query, hit the sparql parsing server and return the json.

        	The function assumes a sparql parsing server running on http://localhost:3000
        	To run the server simply find the file app.js and then execute nodejs ap.js
        '''
        query = {}
        query['query'] = sparql_query
        r = requests.post("http://localhost:3000/parsesparql", json=query)
        return r.json()

    def handle_count(self,sparql_query):
        '''
            SPARQL parsing serer as well as Qald Parser(new name) only supports count if it is of form
                 (COUNT(?uri) as ?uri) with uri being the variable
                 or
                 (COUNT(?x) as ?x) with x being the variable.

                 No other variable is supported currently.
        :param sparql_query: a sparql query with above described structure
        :return: sparql query in a cleaner format (COUNT(?uri) as ?uri),(COUNT(?x) as ?x)
        '''
        sparql_query = sparql_query.replace('COUNT(?uri)' , '(COUNT(?uri) as ?uri)')
        sparql_query = sparql_query.replace('COUNT(?x)' , '(COUNT(?uri) as ?x)')
        return sparql_query

    def dataset_preparation_time(self,_data_node):
        parsed_sparql = self.parse_sparql(_data_node['sparql_query'])
        print('parsed sparql is ', parsed_sparql)
        path, entity, constraints = qp.get_true_path(parsed_sparql,_data_node['sparql_query'])
        print('entity is ', entity)
        hop1,hop2  = self.create_subgraph.subgraph(entity,_data_node['corrected_question'],self.relation_file,_use_blacklist=True,_qald=self.qald)
        return hop1,hop2,path,entity,constraints


