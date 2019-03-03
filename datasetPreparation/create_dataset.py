'''
    >Read node information
    >Send each SPARQL to the SPARQL parsing service
    >Send each response of the sparql parsing service to retrieve constraints, true paths and entites
    >Send the entity to the entity_subgraph class to retrive the required subgraph
    >Pack it in specific manner
'''
import os
import requests
import traceback

import qald_parser as qp
import utils.natural_language_utilities as nlutils
from datasetPreparation import entity_subgraph as es
from utils import dbpedia_interface as db_interface
from datasetPreparation import rdf_candidates as rdfc

os.environ['NO_PROXY'] = 'localhost'


class CreateDataNode():
    def __init__(self,_predicate_blacklist,_relation_file,_qald=False):
        self.dbp = db_interface.DBPedia(caching=True)
        self.relation_file = _relation_file
        self.predicate_blacklist = _predicate_blacklist
        self.qald = _qald
        self.create_subgraph = es.CreateSubgraph(self.dbp, self.predicate_blacklist, self.relation_file, qald=_qald)

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
            hop1_sf = [[r[0],nlutils.get_label_via_parsing(r[1])] for r in hop1]
            path_sf = [path[0],nlutils.get_label_via_parsing(path[1])]
            try:
                index = hop1_sf.index(path_sf)
                hop1.pop(index)
                return hop1,hop2,True
            except ValueError:
                # print(hop1_sf[0],path_sf[0])
                print(traceback.print_exc())
                return hop1,hop2,False
        else:
            try:
                hop2_sf = [[r[0], nlutils.get_label_via_parsing(r[1]), r[2], nlutils.get_label_via_parsing(r[3]) ] for r in hop2]
                path_sf = [path[0], nlutils.get_label_via_parsing(path[1]),path[2],nlutils.get_label_via_parsing(path[3])]
                index = hop2_sf.index(path_sf)
                hop2.pop(index)
                return hop1,hop2,True
            except ValueError:
                # print(hop2_sf[0], path_sf[0])
                print(traceback.print_exc())
                return hop1,hop2,False

    def remove_truepath_from_path_constraint(self,true_constraint,constraint_list):
        '''

        :param constraint:
        :param uri_const:
        :param x_const:
        :return:
        '''
        try:
            const_list_sf = [nlutils.get_label_via_parsing(r) for r in constraint_list]
            true_constraint_sf = nlutils.get_label_via_parsing(true_constraint)
            index = const_list_sf.index(true_constraint_sf)
            constraint_list.pop(index)
            return constraint_list, True
        except ValueError:
            return constraint_list, False


    def generate_rdf_constraint(self,path,entity,dbp,constraint):
        '''

        :param path:
        :param entity: []
        :param dbp:
        :param constraint:
        :return:


        Constraints flag can be of three stage
            - True - true path was found in false path
            - False - no true path was found
            - 'no constraints' - has no constraints
        '''
        if '?uri' in constraint or '?x' in constraint:
            x_const, uri_const = rdfc.generate_rdf_candidates(path=path, topic_entity=entity, dbp=dbp)
            if '?uri' in constraint:
                uri_const,flag = self.remove_truepath_from_path_constraint(
                    true_constraint=constraint['?uri'],constraint_list=uri_const)
            else:
                x_const,flag = self.remove_truepath_from_path_constraint(
                    true_constraint=constraint['?x'], constraint_list=x_const
                )
            rdf_candidate = {
                'constraint_flag':flag,
                'candidates':{}
                             }
            rdf_candidate['candidates']['uri'] = uri_const
            rdf_candidate['candidates']['x'] = x_const
            return rdf_candidate
        else:
            rdf_candidate = {
                'constraint_flag': 'no constraints',
                'candidates': {}
            }
            rdf_candidate['candidates']['uri'] = []
            rdf_candidate['candidates']['x'] = []
            return rdf_candidate


    def dataset_preparation_time(self,_data_node,rdf=True):
        '''

            There are 4 cases to handle for returning
                > Constraints as well as positive path were found in data generation - Perfect
                > Constraints found but no positive path found in data generation -
                > No constraints found but positive path found in data generation -
                > No constraints or positive path found.

         :param self:
         :param _data_node:
         :param rdf - If true, also finds rdf constraints on the given true path.
         :return:

        '''

        data = {
            'node' : _data_node,
            'parsed_sparql' : '',
            'path':[],
            'entity':[],
            'constraints':{},
            'updated_sparql':'',
            'hop1':[],
            'hop2':[],
            'error_flag':{
                'path_found_in_data_generated':False,
                'constraint_found_in_data_generated':False,
            },
            'rdf_constraint' : {}
        }

        sparql = self.handle_count(_data_node['sparql_query'])
        data['updated_sparql'] = sparql

        parsed_sparql = self.parse_sparql(sparql_query=sparql)
        data['parsed_sparql'] = parsed_sparql

        '''
           constraints - {'count': True, '?uri': 'http://dbpedia.org/ontology/TelevisionShow'}
        '''

        path, entity, constraints = qp.get_true_path(parsed_sparql,_data_node['sparql_query'])
        path = self.handle_path(path)
        data['entity'],data['constraints'] = entity,constraints
        data['path'] = path


        hop1,hop2  = self.create_subgraph.subgraph\
            (entity,_data_node['corrected_question'],self.relation_file,_use_blacklist=True,_qald=self.qald)



        hop1,hop2,path_found_in_data_generated = self.remove_truepath_from_paths(path,hop1,hop2)
        data['hop1'],data['hop2'],data['error_flag']['path_found_in_data_generated'] =\
            hop1,hop2,path_found_in_data_generated

        '''
            rdf type generation
        '''
        if rdf:
            rdf_candidates = self.generate_rdf_constraint(path=path,entity=entity,dbp=self.dbp,constraint=constraints)
            data['rdf_constraint'] = rdf_candidates
            if rdf_candidates['constraint_flag']:
                data['error_flag']['constraint_found_in_data_generated'] = True
            else:
                data['error_flag']['constraint_found_in_data_generated'] = False

        return data