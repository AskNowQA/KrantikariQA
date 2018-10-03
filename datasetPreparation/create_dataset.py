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

    def handle_path(self,path):
        '''
         Currently path output is --> ['-http://dbpedia.org/property/mother', '+http://dbpedia.org/property/spouse']
        Shift to standard semantics --> ['-','http://dbpedia.org/property/mother', '+', 'http://dbpedia.org/property/spouse']

        :param path: ['-http://dbpedia.org/property/mother', '+http://dbpedia.org/property/spouse']
        :return: ['-','http://dbpedia.org/property/mother', '+', 'http://dbpedia.org/property/spouse']
        '''
        new_path = []
        for p in path:
            new_path.append(p[0])
            new_path.append(p[1:])

        return new_path

    def remove_truepath_from_paths(self,path,hop1,hop2):
        '''

            Since the dataset generated would contain true paths in the false path, one needs to remove it
            before using it.
            if no true path is found return False

            path -> ['-','http://dbpedia.org/property/mother', '+', 'http://dbpedia.org/property/spouse']
        :return:
        '''

        if len(path) == 2:
            try:
                index = hop1.index(path)
                hop1.pop(index)
                return hop1,hop2
            except ValueError:
                return False,False
        else:
            try:
                index = hop2.index(path)
                hop2.pop(index)
                return hop1,hop2
            except ValueError:
                return False,False

    def dataset_preparation_time(self,_data_node):
        _data_node['sparql_query'] = self.handle_count(_data_node['sparql_query'])
        parsed_sparql = self.parse_sparql(_data_node['sparql_query'])
        '''
            constraints - {'count': True, '?uri': 'http://dbpedia.org/ontology/TelevisionShow'}
        '''
        path, entity, constraints = qp.get_true_path(parsed_sparql,_data_node['sparql_query'])
        hop1,hop2  = self.create_subgraph.subgraph\
            (entity,_data_node['corrected_question'],self.relation_file,_use_blacklist=True,_qald=self.qald)

        path = self.handle_path(path)

        new_hop1,new_hop2 = self.remove_truepath_from_paths(path,hop1,hop2)
        path_found_in_data_generated = True
        if new_hop1 and new_hop2:
            return new_hop1, new_hop2, path, entity, constraints,path_found_in_data_generated
        else:
            path_found_in_data_generated = False
            return hop1,hop2,path,entity,constraints,path_found_in_data_generated