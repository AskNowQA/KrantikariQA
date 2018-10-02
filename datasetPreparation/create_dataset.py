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

    def runtime(self,_data_node):
        parsed_sparql = self.parse_sparql(_data_node['sparql_query'])
        path, entity, constraints = qp.get_true_path(parsed_sparql,_data_node['sparql_query'])
        hop1,hop2  = self.create_subgraph.subgraph([entity],_data_node['question'],self.relation_file,_use_blacklist=True,_qald=self.qald)



