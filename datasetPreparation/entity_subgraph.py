'''

    Takes in a entity as input and retrives a two hop sub graph around it by querying DBpedia which doesn't include any literal.

'''
import numpy as np
from utils import embeddings_interface
from utils import natural_language_utilities as nlutils
import traceback

class CreateSubgraph:
    def __init__(self,_dbpedia_interface,_predicate_blacklist,relation_file, qald=False):

        self.K_1HOP_GLOVE = 200
        self.K_1HOP_MODEL = 5
        self.K_2HOP_GLOVE = 2000
        self.K_2HOP_MODEL = 5
        self.EMBEDDING = "ulmfit"
        self.TRAINING = True # for older system
        self.qald = qald #Dataset specific stuff. As the predicate blacklist would be differently handled when compared to LC-QuAD.

        # Useful objects
        self.dbp = _dbpedia_interface

        #Static resources
        '''
            A dictionary of relation with key being relation and value being id and some aux data.
            The aux data wouldpaths_hop1_id not be useful for the current use case.
        '''
        self.relation = relation_file

        self.predicate_blacklist = _predicate_blacklist

    def similar_predicates(self,_v_qt, _predicates, _return_indices=False, _k=5):
        """
            Function used to tokenize the question and compare the tokens with the predicates.
            Then their top k are selected.
        """

        # If there are no predicates
        if len(_predicates) == 0:
            return np.asarray([]) if _return_indices else []


            # Declare a similarity array
        similarity_arr = np.zeros(len(_predicates))

        # Fill similarity array
        for i in range(len(_predicates)):
            try:
                p = _predicates[i].decode("utf-8")
            except:
                p = _predicates[i]
            v_p = np.mean(embeddings_interface.vectorize(nlutils.tokenize(p), _embedding=self.EMBEDDING).astype(np.float), axis=0)

            # If either of them is a zero vector, the cosine is 0.\
            if np.sum(v_p) == 0.0 or np.sum(_v_qt) == 0.0 or p.strip() == "":
                similarity_arr[i] = np.float64(0.0)
                continue
            try:
                # Cos Product
                similarity_arr[i] = np.dot(v_p, _v_qt) / (np.linalg.norm(v_p) * np.linalg.norm(_v_qt))
            except:
                traceback.print_exc()

        # Find the best scoring values for every path
        # Sort ( best match score for each predicate) in descending order, and choose top k
        argmaxes = np.argsort(similarity_arr, axis=0)[::-1][:_k]

        if _return_indices:
            return argmaxes

        # Use this to choose from _predicates and return
        return [_predicates[i] for i in argmaxes]

    @staticmethod
    def filter_predicates(_predicates, predicate_blacklist,_use_blacklist=True, _only_dbo=False, _qald=False):
        """
            Function used to filter out predicates based on some logic
                - use a blacklist/whitelist @TODO: Make and plug one in.
                - only use dbo predicates, even.

        :param _predicates: A list of strings (uri of predicates)
        :param _use_blacklist: bool
        :param _only_dbo: bool
        :param _qald: Dataset specifc stuff as qald allows 'http://purl.org/dc/terms/subject' and 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' as primary relation.
        :return: A list of strings (uri)


        Note that predicate blacklist for QALD would be different when compared to LC-QuAD
        """

        if _use_blacklist and not _qald:
            _predicates = [x for x in _predicates
                           if x not in predicate_blacklist]

        if _only_dbo and not _qald:
            _predicates = [x for x in _predicates
                           if x.startswith('http://dbpedia.org/ontology')
                           or x.startswith('dbo:')]

        if _qald:
            if _use_blacklist:
                _predicates = [x for x in _predicates
                               if x not in predicate_blacklist]
            predicates_new = []
            for x in _predicates:
                if x.startswith('http://purl.org/dc/terms/subject'):
                    predicates_new.append(x)
                if _only_dbo:
                    if x.startswith('http://dbpedia.org/ontology') or x.startswith('dbo:') or \
                            x.startswith('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
                        predicates_new.append(x)
                else:
                    if not x.startswith('http://purl.org/dc/terms/'):
                        predicates_new.append(x)
            _predicates = predicates_new

        # Filter out uniques
        # _predicates = _predicates

        return _predicates

    def check_path_for_filter(self,path,predicate_blacklist,_use_blacklist=True, _only_dbo=False, _qald=False):
        '''


        :param path: ['-', 'http://dbpedia.org/property/mother', '+','http://dbpedia.org/property/regent']
        :param predicate_blacklist: a list of black list path
        :param _use_blacklist: Same as filter predicate
        :param _only_dbo: Same as filter predicate
        :param _qald: Same as filter predicate
        :return:

            path is ['-', 'http://dbpedia.org/property/mother', '+','http://dbpedia.org/property/regent']
            It checks if every relation in the path is part of blacklist or not. If no return True else fasle
        '''
        if len(path) == 2:
            rel = [path[1]]
            if len(self.filter_predicates(rel, predicate_blacklist, _use_blacklist=True, _only_dbo=False,
                                          _qald=False)) == 1:
                return True
        else:
            rel = [path[1],path[-1]]
            if len(self.filter_predicates(rel, predicate_blacklist,_use_blacklist=True, _only_dbo=False, _qald=False)) == 2:
                return True
        return False


    @staticmethod
    def get_two_topic_entity_paths(SPARQL, te1, te2, id, dbp):
        if id == 1:
            temp = {}
            temp['te1'] = te1
            temp['te2'] = te2
            answer = dbp.get_answer(SPARQL)  # -,+
            data_temp = []
            for i in range(len(answer['r1'])):
                data_temp.append(['-', answer['r1'][i].decode("utf-8"), "+", answer['r2'][i].decode("utf-8")])
            temp['path'] = data_temp
            return temp
        if id == 2:
            temp = {}
            temp['te1'] = te1
            temp['te2'] = te2
            answer = dbp.get_answer(SPARQL)  # -,+
            data_temp = []
            for i in range(len(answer['r1'])):
                data_temp.append(['+', answer['r1'][i].decode("utf-8"), "-", answer['r2'][i].decode("utf-8")])
            temp['path'] = data_temp
            return temp
        if id == 3:
            temp = {}
            temp['te1'] = te1
            temp['te2'] = te2
            answer = dbp.get_answer(SPARQL)  # -,+
            data_temp = []
            for i in range(len(answer['r1'])):
                data_temp.append(['+', answer['r1'][i].decode("utf-8"), "+", answer['r2'][i].decode("utf-8")])
            temp['path'] = data_temp
            return temp
        if id == 4:
            temp = {}
            temp['te1'] = te1
            temp['te2'] = te2
            answer = dbp.get_answer(SPARQL)  # -,+
            data_temp = []
            for i in range(len(answer['r1'])):
                data_temp.append(['+', answer['r1'][i].decode("utf-8")])
            temp['path'] = data_temp
            return temp
        if id == 5:
            temp = {}
            temp['te1'] = te1
            answer = dbp.get_answer(SPARQL)  # -,+
            data_temp = []
            for i in range(len(answer['r1'])):
                data_temp.append(['+', answer['r1'][i].decode("utf-8")])
            temp['path'] = data_temp
            return temp
        if id == 6:
            temp = {}
            temp['te1'] = te1
            answer = dbp.get_answer(SPARQL)  # -,+
            data_temp = []
            for i in range(len(answer['r1'])):
                data_temp.append(['-', answer['r1'][i].decode("utf-8")])
            temp['path'] = data_temp
            return temp

    def legacy_get_hop2_subgraph(self, _entity, _predicate, dbp, _right=True):
        """
            Function fetches the 2hop subgraph around this entity, and following this predicate
        :param _entity: str: URI of the entity around which we need the subgraph
        :param _predicate: str: URI of the predicate with which we curtail the 2hop graph (and get a tangible number of ops)
        :param _right: Boolean: True -> _predicate is on the right of entity (outgoing), else left (incoming)

        :return: List, List: ight (outgoing), left (incoming) preds
        """
        # Get the entities which will come out of _entity +/- _predicate chain
        intermediate_entities = self.dbp.get_entity(_entity, [_predicate], _right)

        # Filter out the literals, and keep uniques.
        intermediate_entities = list(
            set([x for x in intermediate_entities if x.startswith('http://dbpedia.org/resource')]))

        if len(intermediate_entities) > 100:
            intermediate_entities = intermediate_entities[0:100]
        left_predicates, right_predicates = [], []  # Places to store data.

        for entity in intermediate_entities:
            temp_r, temp_l = self.dbp.get_properties(_uri=entity, label=False)
            left_predicates += temp_l
            right_predicates += temp_r

        return list(set([r.decode("utf-8") for r in right_predicates])), list(
                set([l.decode("utf-8") for l in left_predicates]))

    def get_hop2_subgraph(self,_entity,_predicate,dbp,_right=True):
        '''



        :param _entity: central entity
        :param _predicate: a predicate after which one needs the subgraph
        :return: a set of outgoing and incoming predicates with respect to that of given _predicate
        '''
        right_properties,left_properties = dbp.get_hop2_subgraph(str(_entity),_predicate,right=_right)
        # print("len in the function ", len(right_properties),len(left_properties))
        return list(set([r.decode("utf-8")  for r in right_properties])),list(set([l.decode("utf-8") for l in left_properties]))

    @classmethod
    def two_topic_entity(cls, te1, te2, dbp):
        '''
            The reason to handle two topic entity differtly is that, having two entites can help in
            drastically reducing search space.
            There are three ways to fit the set of te1,te2 and r1,r2
             > SELECT DISTINCT ?uri WHERE { ?uri <%(e_to_e_out)s> <%(e_out_1)s> . ?uri <%(e_to_e_out)s> <%(e_out_2)s>}
             > SELECT DISTINCT ?uri WHERE { <%(e_in_1)s> <%(e_in_to_e_1)s> ?uri. <%(e_in_2)s> <%(e_in_to_e_2)s> ?uri}
             > SELECT DISTINCT ?uri WHERE { <%(e_in_1)s> <%(e_in_to_e_1)s> ?uri. ?uri <%(e_in_2)s> <%(e_in_to_e_2)s> }
        '''
        if te1[0] != "<":
            te1 = "<" + te1 + ">"
        if te2[0] != "<":
            te2 = "<" + te2 + ">"

        data = []
        SPARQL1 = '''SELECT DISTINCT ?r1 ?r2 WHERE { ?uri ?r1 %(te1)s. ?uri ?r2 %(te2)s . } '''
        SPARQL2 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s ?r1 ?uri.  %(te2)s ?r2 ?uri . } '''
        SPARQL3 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s ?r1 ?uri.  ?uri ?r2 %(te2)s . } '''
        SPARQL4 = '''SELECT DISTINCT ?r1 WHERE { %(te1)s ?r1 %(te2)s . } '''

        SPARQL1 = SPARQL1 % {'te1': te1, 'te2': te2}
        SPARQL2 = SPARQL2 % {'te1': te1, 'te2': te2}
        SPARQL3 = SPARQL3 % {'te1': te1, 'te2': te2}
        SPARQL4 = SPARQL4 % {'te1': te1, 'te2': te2}

        data.append(cls.get_two_topic_entity_paths(SPARQL1, te1, te2, 1, dbp))
        # data.append(cls.get_something(SPARQL1, te2, te1, 1, dbp))
        data.append(cls.get_two_topic_entity_paths(SPARQL2, te1, te2, 2, dbp))
        # data.append(cls.get_something(SPARQL2, te2, te1, 2, dbp))
        data.append(cls.get_two_topic_entity_paths(SPARQL3, te1, te2, 3, dbp))
        temp = cls.get_two_topic_entity_paths(SPARQL4, te1, te2, 4, dbp)
        print(len(temp['path']), " is the length of the path")
        data.append(temp)

        return data

    # data.append(cls.get_something(SPARQL3, te2, te1, 3, dbp))
    def subgraph(self,_entities,_question,_relations,_use_blacklist=True,_qald=False):
        '''


        :param _entities: A list of entity for which the sub-graph needs to be retrived with first being the topic entity.
        :param _relations: check constructor --> static resource section.
        :return: Subgraph

        Even though the _entities in theory can have multiple entity, this function only works for single entity. Two entity subgraph is handled in a different manner.

        '''

        # Two state fail macros
        NO_PATHS = False
        NO_PATHS_HOP2 = False

        #Note that for now two length entitypaths_hop2_uri has not been implemented. The two entity sub graph is handled in a different manner.

        # Tokenize question
        qt = nlutils.tokenize(_question, _remove_stopwords=False)

        # Vectorize question
        # v_qt = " "
        v_qt = np.mean(embeddings_interface.vectorize(qt, _embedding=self.EMBEDDING).astype(np.float), axis=0)

        if len(_entities) == 1:

            # Get 1-hop subgraph around the entity
            right_properties, left_properties = self.dbp.get_properties(_uri=_entities[0], label=False)
            right_properties, left_properties = list(set([r.decode("utf-8") for r in right_properties])),list(set([l.decode("utf-8")for l in left_properties]))

            right_properties = list(set(self.filter_predicates(right_properties, predicate_blacklist=self.predicate_blacklist,_use_blacklist=_use_blacklist, _only_dbo=_qald,
                                                      _qald=_qald)))
            left_properties = list(set(self.filter_predicates(left_properties, predicate_blacklist=self.predicate_blacklist,_use_blacklist=_use_blacklist, _only_dbo=_qald, _qald=_qald)))



            #Converting all of them to the relation id space as defined by _relations. Also if the relation doesn't exists in
            #the relation space it is automatically removed.

            # If no paths are returned  by the dbpedia interface.
            if not right_properties and not left_properties :
                NO_PATHS = True
                return None

            # We need surface form to find the most similar predicates. We do this to reduce search space and create higher quality negative examples.
            right_properties_sf = [self.dbp.get_label(x ) for x in right_properties]
            left_properties_sf = [self.dbp.get_label(x) for x in left_properties]

            # WORD-EMBEDDING FILTERING
            right_properties_filter_indices = self.similar_predicates(_v_qt=v_qt,_predicates=right_properties_sf,
                                                                      _return_indices=True,
                                                                      _k=self.K_1HOP_GLOVE)
            left_properties_filter_indices = self.similar_predicates(_v_qt=v_qt,_predicates=left_properties_sf,
                                                                     _return_indices=True,
                                                                     _k=self.K_1HOP_GLOVE)

            # Generate their URI counterparts
            right_properties_filtered_uri = [right_properties[i] for i in right_properties_filter_indices]
            left_properties_filtered_uri = [left_properties[i] for i in left_properties_filter_indices]

            paths_hop1_uri = [['+', _p] for _p in right_properties_filtered_uri]
            paths_hop1_uri += [['-', _p] for _p in left_properties_filtered_uri]

            left_properties_filtered, right_properties_filtered = left_properties_filtered_uri, right_properties_filtered_uri

            """
            				2 - Hop COMMENCES

            				Note: Switching to LC-QuAD nomenclature hereon. Refer to /resources/nomenclature.png
            """

            e_in_in_to_e_in = {}
            e_in_to_e_in_out = {}
            e_out_to_e_out_out = {}
            e_out_in_to_e_out = {}

            for pred in right_properties_filtered:
                temp_r, temp_l = self.get_hop2_subgraph(_entity=_entities[0], _predicate=pred,dbp=self.dbp, _right=True)
                e_out_to_e_out_out[pred] = self.filter_predicates(temp_r,predicate_blacklist=self.predicate_blacklist, _use_blacklist=True, _only_dbo=_qald,
                                                                  _qald=_qald)
                e_out_in_to_e_out[pred] = self.filter_predicates(temp_l, predicate_blacklist=self.predicate_blacklist,_use_blacklist=True, _only_dbo=_qald,
                                                                 _qald=_qald)

            for pred in left_properties_filtered:
                temp_r, temp_l = self.get_hop2_subgraph(_entity=_entities[0], _predicate=pred,dbp=self.dbp, _right=False)
                e_in_to_e_in_out[pred] = self.filter_predicates(temp_r, predicate_blacklist=self.predicate_blacklist,_use_blacklist=True, _only_dbo=_qald,
                                                                _qald=_qald)
                e_in_in_to_e_in[pred] = self.filter_predicates(temp_l, predicate_blacklist=self.predicate_blacklist,_use_blacklist=True, _only_dbo=_qald,
                                                               _qald=_qald)

            # Get their surface forms, maintain a key-value store
            sf_vocab = {}
            for key in e_in_in_to_e_in.keys():
                for uri in e_in_in_to_e_in[key]:
                    sf_vocab[uri] = self.dbp.get_label(uri)
            for key in e_in_to_e_in_out.keys():
                for uri in e_in_to_e_in_out[key]:
                    sf_vocab[uri] = self.dbp.get_label(uri)
            for key in e_out_to_e_out_out.keys():
                for uri in e_out_to_e_out_out[key]:
                    sf_vocab[uri] = self.dbp.get_label(uri)
            for key in e_out_in_to_e_out.keys():
                for uri in e_out_in_to_e_out[key]:
                    sf_vocab[uri] = self.dbp.get_label(uri)

            # Flatten the four kind of predicates, and use their surface forms.
            e_in_in_to_e_in_sf = [sf_vocab[x] for uris in e_in_in_to_e_in.values() for x in uris]
            e_in_to_e_in_out_sf = [sf_vocab[x] for uris in e_in_to_e_in_out.values() for x in uris]
            e_out_to_e_out_out_sf = [sf_vocab[x] for uris in e_out_to_e_out_out.values() for x in uris]
            e_out_in_to_e_out_sf = [sf_vocab[x] for uris in e_out_in_to_e_out.values() for x in uris]

            # WORD-EMBEDDING FILTERING
            e_in_in_to_e_in_filter_indices = self.similar_predicates(_v_qt=v_qt,_predicates=e_in_in_to_e_in_sf,
                                                                     _return_indices=True,
                                                                     _k=self.K_2HOP_GLOVE)
            e_in_to_e_in_out_filter_indices = self.similar_predicates(_v_qt=v_qt,_predicates=e_in_to_e_in_out_sf,
                                                                      _return_indices=True,
                                                                      _k=self.K_2HOP_GLOVE)
            e_out_to_e_out_out_filter_indices = self.similar_predicates(_v_qt=v_qt,_predicates=e_out_to_e_out_out_sf,
                                                                        _return_indices=True,
                                                                        _k=self.K_2HOP_GLOVE)
            e_out_in_to_e_out_filter_indices = self.similar_predicates(_v_qt=v_qt,_predicates=e_out_in_to_e_out_sf,
                                                                       _return_indices=True,
                                                                       _k=self.K_2HOP_GLOVE)

            # Impose these indices to generate filtered predicate list.
            e_in_in_to_e_in_filtered = [e_in_in_to_e_in_sf[i] for i in e_in_in_to_e_in_filter_indices]
            e_in_to_e_in_out_filtered = [e_in_to_e_in_out_sf[i] for i in e_in_to_e_in_out_filter_indices]
            e_out_to_e_out_out_filtered = [e_out_to_e_out_out_sf[i] for i in e_out_to_e_out_out_filter_indices]
            e_out_in_to_e_out_filtered = [e_out_in_to_e_out_sf[i] for i in e_out_in_to_e_out_filter_indices]

            # Use them to make a filtered dictionary of hop1: [hop2_filtered] pairs
            e_in_in_to_e_in_filtered_subgraph = {}
            for x in e_in_in_to_e_in_filtered:
                for uri in sf_vocab.keys():
                    if x == sf_vocab[uri]:

                        # That's the URI. Find it's 1-hop Pred.
                        for hop1 in e_in_in_to_e_in.keys():
                            if uri in e_in_in_to_e_in[hop1]:

                                # Now we found a matching :sweat:
                                try:
                                    e_in_in_to_e_in_filtered_subgraph[hop1].append(uri)
                                except KeyError:
                                    e_in_in_to_e_in_filtered_subgraph[hop1] = [uri]

            e_in_to_e_in_out_filtered_subgraph = {}
            for x in e_in_to_e_in_out_filtered:
                for uri in sf_vocab.keys():
                    if x == sf_vocab[uri]:

                        # That's the URI. Find it's 1-hop Pred.get_label
                        for hop1 in e_in_to_e_in_out.keys():
                            if uri in e_in_to_e_in_out[hop1]:

                                # Now we found a matching :sweat:
                                try:
                                    e_in_to_e_in_out_filtered_subgraph[hop1].append(uri)
                                except KeyError:
                                    e_in_to_e_in_out_filtered_subgraph[hop1] = [uri]

            e_out_to_e_out_out_filtered_subgraph = {}
            for x in e_out_to_e_out_out_filtered:
                for uri in sf_vocab.keys():
                    if x == sf_vocab[uri]:
                        # That's the URI. Find it's 1-hop Pred.
                        for hop1 in e_out_to_e_out_out.keys():
                            if uri in e_out_to_e_out_out[hop1]:

                                # Now we found a matching :sweat:
                                try:
                                    e_out_to_e_out_out_filtered_subgraph[hop1].append(uri)
                                except KeyError:
                                    e_out_to_e_out_out_filtered_subgraph[hop1] = [uri]

            e_out_in_to_e_out_filtered_subgraph = {}
            for x in e_out_in_to_e_out_filtered:
                for uri in sf_vocab.keys():
                    if x == sf_vocab[uri]:

                        # That's the URI. Find it's 1-hop Pred.
                        for hop1 in e_out_in_to_e_out.keys():
                            if uri in e_out_in_to_e_out[hop1]:

                                # Now we found a matching :sweat:
                                try:
                                    e_out_in_to_e_out_filtered_subgraph[hop1].append(uri)
                                except KeyError:
                                    e_out_in_to_e_out_filtered_subgraph[hop1] = [uri]


            paths_hop2_uri = []
            for key in e_in_in_to_e_in_filtered_subgraph.keys():
                for r2 in e_in_in_to_e_in_filtered_subgraph[key]:
                    path_uri = ['-', key, '-', r2]
                    paths_hop2_uri.append(path_uri)


            for key in e_in_to_e_in_out_filtered_subgraph.keys():
                for r2 in e_in_to_e_in_out_filtered_subgraph[key]:

                    path_uri = ['-', key, '+', r2]
                    paths_hop2_uri.append(path_uri)


            for key in e_out_to_e_out_out_filtered_subgraph.keys():
                for r2 in e_out_to_e_out_out_filtered_subgraph[key]:
                    path_uri = ['+', key, '+', r2]
                    paths_hop2_uri.append(path_uri)


            for key in e_out_in_to_e_out_filtered_subgraph.keys():
                for r2 in e_out_in_to_e_out_filtered_subgraph[key]:
                    path_uri = ['+', key, '-', r2]
                    paths_hop2_uri.append(path_uri)


        if len(_entities) == 2:
            results = self.two_topic_entity(_entities[0], _entities[1], self.dbp)

            # return [], results
            #Filtering out blacklisted relationships
            all_paths = []
            for node in results:
                all_paths = all_paths + node['path']

            paths_hop2_uri = []
            paths_hop1_uri = []

            for path in all_paths:
                if self.check_path_for_filter(path,predicate_blacklist=self.predicate_blacklist
                        ,_use_blacklist=True, _only_dbo=_qald, _qald=_qald):
                    if len(path) > 2:
                        paths_hop2_uri.append(path)
                    else:
                        paths_hop1_uri.append(path)


        # #idfiy everything and return.
        # paths_hop1_id = []
        # for p in paths_hop1_uri:
        #     try:
        #         paths_hop1_id.append([p[0],_relations[p[1]][0]])
        #     except:
        #         continue
        #
        # paths_hop2_id = []
        # for p in paths_hop2_uri:
        #     try:
        #         paths_hop2_id.append([p[0],_relations[p[1]][0],p[2],_relations[p[3]][0]])
        #     except:
        #         continue
        return [list(x) for x in set(tuple(x) for x in paths_hop1_uri)],[list(x) for x in set(tuple(x) for x in paths_hop2_uri)]