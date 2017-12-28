"""

    Author:     saist1993
    Desc:       Used to pre-process training data. Uses LC-QuAD data as the input and creates parser.py friendly JSONs.


    The assumption is that the constraint is either on uri or x . The constraints can be count or type.

    @TODO: Check for the outgoing and incoming conventions wrt True and False
"""

import os
import json
import traceback
from pprint import pprint

# Custom files
import utils.dbpedia_interface as db_interface
import utils.embeddings_interface as sim


'''
    MACROS
'''
K_HOP_1 = 5                                 # Selects the number of relations in the first hop in the right direction
K_HOP_2 = 5                                 # Selects the number of relations in second hop in the right direction
K_HOP_1_u = 2                               # Selects the number of relations in second hop in the wrong direction
K_HOP_2_u = 2                               # Selects the number of relations in second hop in the wrong direction
PASSED = False
WRITE_INTERVAL = 10                         # Interval for periodic write in a file
OUTPUT_DIR = 'data/preprocesseddata_new_v2'    # Place to store the files


'''
    Global variables
'''
# Summon a dbpedia interface
dbp = db_interface.DBPedia(_verbose=True, caching=False)

skip = 0
relations_stop_word = []
final_answer_dataset = []                   # Used in create_simple_dataset()


# Ensure that a folder already exists in OUTPUT_DIR
try:
    os.makedirs(OUTPUT_DIR)
except OSError:
    print("Folder already exists")


def get_rank_rel(_relationsip_list, rel,_score=False):
    """
        The objective is to rank the relationship using some trivial similarity measure wrt rel
        [[list of outgoing rels],[list of incoming rels]] (rel,True)  'http://dbpedia.org/ontology/childOrganisation'
        Need to verify the function
    """

    # Transforming the list of items into a list of tuple
    score = []
    new_rel_list = []
    outgoing_temp = []
    for rels in _relationsip_list[0]:
        score.append((rels,sim.phrase_similarity(dbp.get_label(rel[0]),dbp.get_label(rels))))
     # print sorted(score, key=lambda score: score[1], reverse=True)
    new_rel_list.append(sorted(score, key=lambda score: score[1],reverse=True))

    score = []
    for rels in _relationsip_list[1]:
        score.append((rels,sim.phrase_similarity(dbp.get_label(rel[0]),dbp.get_label(rels))))
    new_rel_list.append(sorted(score, key=lambda score: score[1],reverse=True))

    final_rel_list = []

    final_rel_list.append([x[0] for x in new_rel_list[0]])
    final_rel_list.append([x[0] for x in new_rel_list[1]])

    # print rel
    # pprint(final_rel_list)
    # raw_input('check')
    if not  _score:
        return final_rel_list
    else:
        return new_rel_list
    # return _relationsip_list


def get_set_list(_list):
    for i in xrange(0,len(_list)):
        _list[i] =list(set(_list[i]))
    return _list


def get_top_k(rel_list,_relation,hop=1):
    # Once the similarity been computed and ranked accordingly, take top k based on some metric.
    # pprint(rel_list)
    if hop == 1:
        if _relation[1]:
            if len(rel_list[0]) >= K_HOP_1:
                rel_list[0] = rel_list[0][:K_HOP_1]
            if len(rel_list[1]) > K_HOP_1_u:
                rel_list[1] = rel_list[1][:K_HOP_1_u]
        else:
            if len(rel_list[0]) >= K_HOP_1_u:
                rel_list[0] = rel_list[0][:K_HOP_1_u]
            if len(rel_list[1]) > K_HOP_1:
                rel_list[1] = rel_list[1][:K_HOP_1]
        return  rel_list
    else:
        if _relation[1]:
            if len(rel_list[0]) >= K_HOP_2:
                rel_list[0] = rel_list[0][:K_HOP_2]
            if len(rel_list[1]) > K_HOP_2_u:
                rel_list[1] = rel_list[1][:K_HOP_2_u]
        else:
            if len(rel_list[0]) >= K_HOP_2_u:
                rel_list[0] = rel_list[0][:K_HOP_2_u]
            if len(rel_list[1]) > K_HOP_2:
                rel_list[1] = rel_list[1][:K_HOP_2]
        return  rel_list


def get_triples(_sparql_query):
    '''
        parses sparql query to return a set of triples
    '''
    parsed = _sparql_query.split("{")
    parsed = [x.strip() for x in parsed]
    triples = parsed[1][:-1].strip()
    triples =  triples.split(". ")
    triples = [x.strip() for x in triples]
    return triples


def get_relationship_hop(_entity, _relation):
    '''
        The objective is to find the outgoing and incoming relationships from the entity at _hop distance.
        :param _entity: the seed entity
        :param _relation: A chain of relation [(rel1,True),(rel2,False)] - True represents a outgoing property while False an incoming property.
        :return: [[set(incoming property)],[set(outgoing property]]
    '''
    entities = [_entity]
    for rel in _relation[0:-1]:
        outgoing = rel[1]
        if outgoing:
            ''' get the objects '''
            temp = [dbp.get_entity(_entity,rel[0],outgoing=True) for ent in entities]
            entities = list(set([item for sublist in temp for item in sublist]))
        else:
            '''get the subjects '''
            temp = [dbp.get_entity(_entity, rel[0], outgoing=False) for ent in entities]
            entities = list(set([item for sublist in temp for item in sublist]))

    #Now we have a set of entites and we need to find all relations going from this relationship and also the final relationship
            #should be a a pert of the returned relationship
    #Find all the outgoing and incoming relationships
    outgoing_relationships = []
    incoming_relationships = []
    for ent in entities:
        rel = dbp.get_properties(ent)
        outgoing_relationships =  outgoing_relationships + list(set(rel[0]))
        incoming_relationships = incoming_relationships + list(set(rel[1]))
    outgoing_relationships = list(set(outgoing_relationships))
    incoming_relationships = list(set(incoming_relationships))
    return [outgoing_relationships,incoming_relationships]


def updated_get_relationship_hop(_entity, _relations):
    '''

        This function gives all the relations after the _relationship chain. The
        difference in this and the get_relationship_hop is that it gives all the relationships from _relations[:-1],
    '''
    entities = [_entity]    #entites are getting pushed here
    for rel in _relations:
        outgoing = rel[1]
        if outgoing:
            ''' get the objects '''
            temp = [dbp.get_entity(ent,rel,outgoing=True) for ent in entities]
            entities = list(set([item for sublist in temp for item in sublist]))
        else:
            '''get the subjects '''
            temp = [dbp.get_entity(ent, rel, outgoing=False) for ent in entities]
            entities = list(set([item for sublist in temp for item in sublist]))
        temp_ent = []
        ''' If after the query we get a literal instead of a resources'''
        for ent in entities:
            if "http://dbpedia.org/resource" in ent:
                temp_ent.append(ent)
        entities = temp_ent
    #Now we have a set of entites and we need to find all relations going from this relationship and also the final relationship
            #should be a a pert of the returned relationship
    #Find all the outgoing and incoming relationships
    outgoing_relationships = []
    incoming_relationships = []
    for ent in entities:
        rel = dbp.get_properties(ent,label=False)
        outgoing_relationships =  outgoing_relationships + list(set(rel[0]))
        incoming_relationships = incoming_relationships + list(set(rel[1]))
    outgoing_relationships = list(set(outgoing_relationships))
    incoming_relationships = list(set(incoming_relationships))
    return [outgoing_relationships,incoming_relationships]


def get_stochastic_relationship_hop(_entity, _relation):
    '''
        The objective is to find the outgoing and incoming relationships from the entity at _hop distance.
        :param _entity: the seed entity
        :param _relation: A chain of relation [(rel1,True),(rel2,False)] - True represents a outgoing property while False an incoming property.
        :return: [[set(incoming property)],[set(outgoing property]]
    '''
    out,incoming =  dbp.get_properties(_entity,_relation[0][0],label=False)

    rel_list = get_set_list(get_top_k(get_rank_rel([out,incoming],_relation[0]),_relation[0]))
    # print rel_list
    '''
        Now with each relation list find the next graph and stochastically prune it.
    '''
    outgoing_relationships = []
    incoming_relationships = []
    for rel in rel_list[0]:
        temp = {}

        # get_set_list(get_top_k(get_rank_rel(updated_get_relationship_hop(_entity, (rel, True)),(rel,True)),(rel,True),hop=2))
        # print updated_get_relationship_hop(_entity, [(rel, True)]), (rel, True)
        # print "******"
        temp[rel] = get_set_list(
            get_top_k(get_rank_rel(updated_get_relationship_hop(_entity, [(rel, True)]), (rel, True)), (rel, True),
                      hop=2))
        # temp[rel] = get_set_list(get_top_k(get_rank_rel(updated_get_relationship_hop(_entity,(rel,True)),(rel,True),hop=2)))
        outgoing_relationships.append(temp)

    for rel in rel_list[1]:
        temp = {}
        temp[rel] = get_set_list(get_top_k(get_rank_rel(updated_get_relationship_hop(_entity, [(rel, False)]),(rel,False)),(rel,False),hop=2))
        incoming_relationships.append(temp)
    return [outgoing_relationships,incoming_relationships]


def get_rdf_type_candidates(sparql,rdf_type=True,constraint = '',count=False):
    '''
        Takes in a SPARQL and then updates the sparql to return rdf type. If rdf_type is flase than it assumes there exists no rdf:type.
        The varaible type needs to be named ?uri and the intermediate varaible needs to be termed x
    :param _entity: takes in a sparql and ch
    :return: classes list
    '''
    print constraint
    print sparql
    URI_Type = ' ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type .'
    X_Type = ' ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type .'
    original_string = 'SELECT DISTINCT ?uri WHERE'
    new_string = 'SELECT DISTINCT ?type WHERE'
    if count:
        sparql = sparql.replace('COUNT(?uri)', '?uri')
    if rdf_type:
        '''
            replace rdf_type sentence with nothing
            ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
        '''
        new_URI_Type = '?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>' + " " + "<" + constraint + ">"
        new_X_Type = '?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>' + " " + "<" + constraint + ">"
        sparql = sparql.replace(new_URI_Type, '')
        sparql = sparql.replace(new_X_Type,'')
    temp_sparql = sparql.replace(original_string, new_string)
    index = temp_sparql.find('}')
    temp_sparql_uri = temp_sparql[:index] + URI_Type + temp_sparql[index:]
    type_uri = dbp.get_answer(temp_sparql_uri)
    type_uri = type_uri[u'type']
    type_x = []
    if '?x' in sparql:
        temp_sparql_x = temp_sparql[:index] + X_Type + temp_sparql[index:]
        type_x = dbp.get_answer(temp_sparql_x)
        type_x = type_x[u'type']
        type_x = [x for x in type_x if
                                  x[:28] in ['http://dbpedia.org/ontology/', 'http://dbpedia.org/property/']]
    type_uri = [x for x in type_uri if
              x[:28] in ['http://dbpedia.org/ontology/', 'http://dbpedia.org/property/']]
    return type_uri,type_x
# print get_rdf_type_candidates(sparql=sparql,rdf_type=True,constraint = constraint)



fo = open('interm_output.txt', 'w+')
debug = True
controller = []


def create_dataset(debug=True,time_limit=False):
    final_data = []
    file_directory = "resources/data_set.json"
    json_data = open(file_directory).read()
    data = json.loads(json_data)
    counter = 0
    skip = 0
    for node in data:
        '''
            For now focusing on just simple question
        '''
        print counter
        counter += 1
        if counter == 40:
            continue
        if skip > 0:
            skip -= 1
            continue
        try:
            if node[u"sparql_template_id"] in [1,301,401,101] and not PASSED: # :
                '''
                    {
                        u'_id': u'9a7523469c8c45b58ec65ed56af6e306',
                        u'corrected_question': u'What are the schools whose city is Reading, Berkshire?',
                        u'sparql_query': u' SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/ontology/city> <http://dbpedia.org/resource/Reading,_Berkshire> } ',
                        u'sparql_template_id': 1,
                        u'verbalized_question': u'What are the <schools> whose <city> is <Reading, Berkshire>?'
                    }
                '''
                data_node = node
                if ". }" not in node[u'sparql_query']:
                    node[u'sparql_query'] = node[u'sparql_query'].replace("}",". }")
                triples = get_triples(node[u'sparql_query'])
                data_node[u'entity'] = []
                data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
                data_node[u'training'] = {}
                data_node[u'training'][data_node[u'entity'][0]] = {}
                data_node[u'training'][data_node[u'entity'][0]][u'rel1'] = [list(set(rel)) for rel in list(dbp.get_properties(data_node[u'entity'][0],label=False))]
                #need to include things here.
                data_node[u'training'][data_node[u'entity'][0]][u'rel2'] = get_stochastic_relationship_hop(
                    data_node[u'entity'][0], [(triples[0].split(" ")[1][1:-1], False), (triples[0].split(" ")[1][1:-1], True)])
                data_node[u'path'] = ["-" + triples[0].split(" ")[1][1:-1]]
                data_node[u'constraints'] = {}
                if node[u"sparql_template_id"] == 301 or node[u"sparql_template_id"] == 401:
                    data_node[u'constraints'] = {triples[1].split(" ")[0]: triples[1].split(" ")[2][1:-1]}
                    if node[u"sparql_template_id"] == 301:
                        value = get_rdf_type_candidates(node[u'sparql_query'],rdf_type=True,constraint=triples[1].split(" ")[2][1:-1],count=False)
                        data_node[u'training']['x'] = value[1]
                        data_node[u'training']['uri'] = value[0]
                else:
                    if node[u"sparql_template_id"] == 1:
                        data_node[u'constraints'] = {}
                        value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                        constraint='', count=False)
                        data_node[u'training']['x'] = value[1]
                        data_node[u'training']['uri'] = value[0]

                if node[u"sparql_template_id"] in [401,101]:
                    data_node[u'constraints'] = {'count' : True}
                    if node[u"sparql_template_id"] == 401:
                        value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                        constraint=triples[1].split(" ")[2][1:-1], count=True)
                        data_node[u'training']['x'] = value[1]
                        data_node[u'training']['uri'] = value[0]
                    if node[u"sparql_template_id"] == 101:
                        value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                        constraint='', count=True)
                        data_node[u'training']['x'] = value[1]
                        data_node[u'training']['uri'] = value[0]
                fo.write(str(data_node))
                fo.write("\n")
                final_data.append(data_node)
                if debug:
                    if data_node['sparql_template_id'] not in controller:
                        pprint(data_node)
                        controller.append(data_node['sparql_template_id'])
            elif node[u"sparql_template_id"] in [2,302,402,102] and not PASSED:
                '''
                    {	u'_id': u'8216e5b6033a407191548689994aa32e',
                        u'corrected_question': u'Name the municipality of Roberto Clemente Bridge ?',
                        u'sparql_query': u' SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Roberto_Clemente_Bridge> <http://dbpedia.org/ontology/municipality> ?uri } ',
                        u'sparql_template_id': 2,
                        u'verbalized_question': u'What is the <municipality> of Roberto Clemente Bridge ?'
                    }
                '''
                #TODO: Verify the 302 template
                data_node = node
                if ". }" not in node[u'sparql_query']:
                    node[u'sparql_query'] = node[u'sparql_query'].replace("}",". }")
                triples = get_triples(node[u'sparql_query'])
                data_node[u'entity'] = []
                data_node[u'entity'].append(triples[0].split(" ")[0][1:-1])
                data_node[u'training'] = {}
                data_node[u'training'][data_node[u'entity'][0]] = {}
                data_node[u'training'][data_node[u'entity'][0]][u'rel1'] =  [list(set(rel)) for rel in list(dbp.get_properties(data_node[u'entity'][0],label=False))]
                data_node[u'training'][data_node[u'entity'][0]][u'rel2'] = get_stochastic_relationship_hop(
                    data_node[u'entity'][0], [(triples[0].split(" ")[1][1:-1], True), (triples[0].split(" ")[1][1:-1], True)])
                data_node[u'path'] = ["+" + triples[0].split(" ")[1][1:-1]]
                data_node[u'constraints'] = {}
                if node[u"sparql_template_id"] == 302 or node[u"sparql_template_id"] == 402:
                    data_node[u'constraints'] = {triples[1].split(" ")[0]: triples[1].split(" ")[2][1:-1]}
                else:
                    data_node[u'constraints'] = {}
                if node[u"sparql_template_id"] in [402,102]:
                    data_node[u'constraints'] = {'count' : True}
                if node[u"sparql_template_id"] == 2:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                    constraint='', count=False)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 102:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                    constraint='', count=True)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 302:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                    constraint=triples[1].split(" ")[2][1:-1], count=False)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 402:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                    constraint=triples[1].split(" ")[2][1:-1], count=True)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                final_data.append(data_node)
                fo.write(str(data_node))
                fo.write("\n")
                if debug:
                    if data_node['sparql_template_id'] not in controller:
                        pprint(data_node)
                        controller.append(data_node['sparql_template_id'])
                        # raw_input()
            elif node[u"sparql_template_id"]  in [3,303,309,9,403,409,103,109] :
                '''
                    {    u'_id': u'dad51bf9d0294cac99d176aba17c0241',
                         u'corrected_question': u'Name some leaders of the parent organisation of the Gestapo?',
                         u'sparql_query': u'SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Gestapo> <http://dbpedia.org/ontology/parentOrganisation> ?x . ?x <http://dbpedia.org/ontology/leader> ?uri  . }',
                         u'sparql_template_id': 3,
                         u'verbalized_question': u'What is the <leader> of the <government agency> which is the <parent organisation> of <Gestapo> ?'}
                '''
                # pprint(node)
                data_node = node
                triples = get_triples(node[u'sparql_query'])
                data_node[u'entity'] = []
                data_node[u'entity'].append(triples[0].split(" ")[0][1:-1])
                rel2 = triples[1].split(" ")[1][1:-1]
                rel1 = triples[0].split(" ")[1][1:-1]
                data_node[u'path'] = ["+" + rel1, "+" + rel2]
                data_node[u'training'] = {}
                data_node[u'training'][data_node[u'entity'][0]] = {}
                data_node[u'training'][data_node[u'entity'][0]][u'rel1'] = [list(set(rel)) for rel in list(dbp.get_properties(data_node[u'entity'][0],label=False))]
                data_node[u'training'][data_node[u'entity'][0]][u'rel2'] = get_stochastic_relationship_hop(data_node[u'entity'][0],[(rel1,True),(rel2,True)])
                if node[u"sparql_template_id"] in [303,309,403,409]:
                    data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[2][1:-1]}
                    if node[u'sparql_template_id'] in [303,309]:
                        value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                        constraint=triples[2].split(" ")[2][1:-1], count=False)
                        data_node[u'training']['x'] = value[1]
                        data_node[u'training']['uri'] = value[0]
                    if node[u'sparql_template_id'] in [403,409]:
                        value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                        constraint=triples[2].split(" ")[2][1:-1], count=True)
                        data_node[u'training']['x'] = value[1]
                        data_node[u'training']['uri'] = value[0]
                else:
                    data_node[u'constraints'] = {}
                if node[u"sparql_template_id"] in [403,409,103,109]:
                    data_node[u'constraints'] = {'count' : True}
                    if node[u'sparql_template_id'] in [103,109]:
                        value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                        constraint='', count=True)
                        data_node[u'training']['x'] = value[1]
                        data_node[u'training']['uri'] = value[0]
                if node[u'sparql_template_id'] in [3, 9]:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                    constraint='', count=False)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                fo.write(str(data_node))
                fo.write("\n")
                final_data.append(data_node)
                if debug:
                    if data_node['sparql_template_id'] not in controller:
                        pprint(data_node)
                        controller.append(data_node['sparql_template_id'])
                        # raw_input()

            elif node[u"sparql_template_id"] in [5,305,405,105,111] and not PASSED:
                '''
                    >Verify this !!
                    {
                        u'_id': u'00a3465694634edc903510572f23b487',
                        u'corrected_question': u'Which party has come in power in Mumbai North?',
                        u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?x <http://dbpedia.org/property/constituency> <http://dbpedia.org/resource/Mumbai_North_(Lok_Sabha_constituency)> . ?x <http://dbpedia.org/ontology/party> ?uri  . }',
                        u'sparql_template_id': 5,
                        u'verbalized_question': u'What is the <party> of the <office holders> whose <constituency> is <Mumbai North (Lok Sabha constituency)>?'
                    }
                '''
                # pprint(node)
                data_node = node
                triples = get_triples(node[u'sparql_query'])
                rel1 = triples[0].split(" ")[1][1:-1]
                rel2 = triples[1].split(" ")[1][1:-1]
                data_node[u'entity'] = []
                data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
                data_node[u'path'] = ["-" + rel1, "+" + rel2]
                data_node[u'training'] = {}
                data_node[u'training'][data_node[u'entity'][0]] = {}
                data_node[u'training'][data_node[u'entity'][0]][u'rel1'] = [list(set(rel)) for rel in
                                                                            list(dbp.get_properties(data_node[u'entity'][0],label=False))]
                data_node[u'training'][data_node[u'entity'][0]][u'rel2'] = get_stochastic_relationship_hop(data_node[u'entity'][0], [(rel1, False), (rel2, True)])
                if node[u"sparql_template_id"] in [305,405] :
                    data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[2][1:-1]}
                else:
                    data_node[u'constraints'] = {}
                if node[u"sparql_template_id"] in [105,405,111]:
                    data_node[u'constraints'] = {'count' : True}
                if node[u"sparql_template_id"] == 5:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                    constraint='', count=False)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 105:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                    constraint='', count=True)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 305:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                    constraint=triples[2].split(" ")[2][1:-1], count=False)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 405:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                    constraint=triples[2].split(" ")[2][1:-1], count=True)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                fo.write(str(data_node))
                fo.write("\n")
                if debug:
                    if data_node['sparql_template_id'] not in controller:
                        pprint(data_node)
                        controller.append(data_node['sparql_template_id'])
                # raw_input()
                final_data.append(data_node)

            elif node[u'sparql_template_id']  == [6, 306, 406, 106] and not PASSED:
                '''
                    {
                        u'_id': u'd3695db03a5e45ae8906a2527508e7c5',
                        u'corrected_question': u'Who have done their PhDs under a National Medal of Science winner?',
                        u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?x <http://dbpedia.org/property/prizes> <http://dbpedia.org/resource/National_Medal_of_Science> . ?uri <http://dbpedia.org/property/doctoralAdvisor> ?x  . }',
                        u'sparql_template_id': 6,
                        u'verbalized_question': u"What are the <scientists> whose <advisor>'s <prizes> is <National Medal of Science>?"
                    }
                '''
                # pprint(node)
                data_node = node
                triples = get_triples(node[u'sparql_query'])
                rel1 = triples[0].split(" ")[1][1:-1]
                rel2 = triples[1].split(" ")[1][1:-1]
                data_node[u'entity'] = []
                data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
                data_node[u'path'] = ["-" + rel1, "-" + rel2]
                data_node[u'training'] = {}
                data_node[u'training'][data_node[u'entity'][0]] = {}
                data_node[u'training'][data_node[u'entity'][0]][u'rel1'] = [list(set(rel)) for rel in
                                                                            list(dbp.get_properties(data_node[u'entity'][0],label=False))]
                data_node[u'training'][data_node[u'entity'][0]][u'rel2'] = get_stochastic_relationship_hop(
                    data_node[u'entity'][0], [(rel1, False), (rel2, False)])
                if node[u"sparql_template_id"] in [306,406]:
                    data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[2][1:-1]}
                else:
                    data_node[u'constraints'] = {}
                if node[u"sparql_template_id"] in [406,106]:
                    data_node[u'constraints'] = {'count' : True}
                # pprint(data_node)
                # raw_input()
                if node[u"sparql_template_id"] == 6:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                    constraint='', count=False)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 106:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=False,
                                                    constraint='', count=True)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 306:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                    constraint=triples[2].split(" ")[2][1:-1], count=False)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                if node[u"sparql_template_id"] == 406:
                    value = get_rdf_type_candidates(node[u'sparql_query'], rdf_type=True,
                                                    constraint=triples[2].split(" ")[2][1:-1], count=True)
                    data_node[u'training']['x'] = value[1]
                    data_node[u'training']['uri'] = value[0]
                fo.write(str(data_node))
                fo.write("\n")
                final_data.append(data_node)
                if debug:
                    if data_node['sparql_template_id'] not in controller:
                        pprint(data_node)
                        controller.append(data_node['sparql_template_id'])

            # print final_data[-1]
            if len(final_data) > WRITE_INTERVAL:
                with open(OUTPUT_DIR+ "/" + str(counter)+ ".json", 'w') as fp:
                    json.dump(final_data, fp)
                final_data = []
        except:
            print traceback.print_exc()
            continue
    with open('remaining.json', 'w') as fp:
        json.dump(final_data, fp)

def test(_entity, _relation):
    out, incoming = dbp.get_properties(_entity, _relation, label=False)
    rel = (_relation, True)
    rel_list = get_rank_rel([out, incoming], rel,score=True)
    # rel_list = get_set_list(get_top_k(get_rank_rel([out,incoming],rel=),rel))
    pprint(rel_list)



def create_simple_dataset():

    file_directory = "resources/data_set.json"
    json_data = open(file_directory).read()
    data = json.loads(json_data)
    counter = 0
    for node in data:
        # print node[u"sparql_template_id"]
        # raw_input("check sparql templated id")
        # pass
        if node[u"sparql_template_id"] in [1] :
            counter = counter + 1
            print counter

            '''
                            {
                                u'_id': u'9a7523469c8c45b58ec65ed56af6e306',
                                u'corrected_question': u'What are the schools whose city is Reading, Berkshire?',
                                u'sparql_query': u' SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/ontology/city> <http://dbpedia.org/resource/Reading,_Berkshire> } ',
                                u'sparql_template_id': 1,
                                u'verbalized_question': u'What are the <schools> whose <city> is <Reading, Berkshire>?'
                            }

            '''
            '''
                >I need answer and the label of the entity
            '''
            answer_data_node = {}
            data_node = node
            triples = get_triples(node[u'sparql_query'])
            data_node[u'entity'] = []
            data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
            data_node[u'training'] = {}
            data_node[u'training'][data_node[u'entity'][0]] = {}
            data_node[u'training'][data_node[u'entity'][0]][u'rel1'] = [list(set(rel)) for rel in list(
                dbp.get_properties(data_node[u'entity'][0],label=False))]
            data_node[u'path'] = ["-" + triples[0].split(" ")[1][1:-1]]
            data_node[u'constraints'] = {}
            if node[u"sparql_template_id"] == 301 or node[u"sparql_template_id"] == 401:
                data_node[u'constraints'] = {triples[1].split(" ")[0]: triples[1].split(" ")[1][1:-1]}
            else:
                data_node[u'constraints'] = {}

            if node[u"sparql_template_id"] in [401, 101]:
                data_node[u'constraints'] = {'count': True}
            final_data.append(data_node)
            # pprint(data_node)
            # pprint("loda")
            answer_data_node['entity'] = dbp.get_label(data_node[u'entity'][0])
            answer_data_node['answer']   = [dbp.get_label(x) for x in dbp.get_answer(data_node[u'sparql_query'])['uri']]
            answer_data_node['question'] = node['corrected_question']
            final_answer_dataset.append(answer_data_node)
            # raw_input()

        elif node[u"sparql_template_id"] in [2]:
            '''
                {	u'_id': u'8216e5b6033a407191548689994aa32e',
                    u'corrected_question': u'Name the municipality of Roberto Clemente Bridge ?',
                    u'sparql_query': u' SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Roberto_Clemente_Bridge> <http://dbpedia.org/ontology/municipality> ?uri } ',
                    u'sparql_template_id': 2,
                    u'verbalized_question': u'What is the <municipality> of Roberto Clemente Bridge ?'
                }
            '''
            counter += 1
            print counter

            #TODO: Verify the 302 template
            answer_data_node = {}
            data_node = node
            triples = get_triples(node[u'sparql_query'])
            data_node[u'entity'] = []
            data_node[u'entity'].append(triples[0].split(" ")[0][1:-1])
            data_node[u'training'] = {}
            data_node[u'training'][data_node[u'entity'][0]] = {}
            data_node[u'training'][data_node[u'entity'][0]][u'rel1'] =  [list(set(rel)) for rel in list(dbp.get_properties(data_node[u'entity'][0],label=False))]
            data_node[u'path'] = ["+" + triples[0].split(" ")[1][1:-1]]
            data_node[u'constraints'] = {}
            if node[u"sparql_template_id"] == 302 or node[u"sparql_template_id"] == 402:
                data_node[u'constraints'] = {triples[1].split(" ")[0]: triples[1].split(" ")[1][1:-1]}
            else:
                data_node[u'constraints'] = {}
            if node[u"sparql_template_id"] in [402,102]:
                data_node[u'constraints'] = {'count' : True}
            final_data.append(data_node)
            answer_data_node['entity'] = dbp.get_label(data_node[u'entity'][0])
            answer_data_node['answer'] = [dbp.get_label(x) for x in dbp.get_answer(data_node[u'sparql_query'])['uri']]
            answer_data_node['question'] = node['corrected_question']
            final_answer_dataset.append(answer_data_node)
            # pprint(final_answer_dataset)
            # raw_input("check at 2")


#TODO: Store as json : final answer dataset

print "datasest call"

def get_something(SPARQL,te1,te2,id):
    if id ==1 :
        temp = {}
        temp['te1'] = te1
        temp['te2'] = te2
        answer = dbp.get_answer(SPARQL)  # -,+
        data_temp = []
        for i in xrange(len(answer['r1'])):
            data_temp.append(['-', answer['r1'][i], "+", answer['r2'][i], '-'])
        temp['path'] = data_temp
        return temp
    if id == 2:
        temp = {}
        temp['te1'] = te1
        temp['te2'] = te2
        answer = dbp.get_answer(SPARQL)  # -,+
        data_temp = []
        for i in xrange(len(answer['r1'])):
            data_temp.append(['+', answer['r1'][i], "+", answer['r2'][i], '-'])
        temp['path'] = data_temp
        return temp
    if id == 3:
        temp = {}
        temp['te1'] = te1
        temp['te2'] = te2
        answer = dbp.get_answer(SPARQL)  # -,+
        data_temp = []
        for i in xrange(len(answer['r1'])):
            data_temp.append(['+', answer['r1'][i], "-", answer['r2'][i], '-'])
        temp['path'] = data_temp
        return temp


def two_topic_entity(te1,te2):
    '''
        There are three ways to fit the set of te1,te2 and r1,r2
         > SELECT DISTINCT ?uri WHERE { ?uri <%(e_to_e_out)s> <%(e_out_1)s> . ?uri <%(e_to_e_out)s> <%(e_out_2)s>}
         > SELECT DISTINCT ?uri WHERE { <%(e_in_1)s> <%(e_in_to_e_1)s> ?uri. <%(e_in_2)s> <%(e_in_to_e_2)s> ?uri}
         > SELECT DISTINCT ?uri WHERE { <%(e_in_1)s> <%(e_in_to_e_1)s> ?uri. ?uri <%(e_in_2)s> <%(e_in_to_e_2)s> }
    '''
    data = []
    SPARQL1 = '''SELECT DISTINCT ?r1 ?r2 WHERE { ?uri ?r1 %(te1)s. ?uri ?r2 %(te2)s . } '''
    SPARQL2 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s ?r1 ?uri.  %(te2)s ?r2 ?uri . } '''
    SPARQL3 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s ?r1 ?uri.  ?uri ?r2 %(te2)s . } '''

    SPARQL1 = SPARQL1 % {'te1': te1, 'te2' : te2}
    SPARQL2 = SPARQL2 % {'te1': te1, 'te2': te2}
    SPARQL3 = SPARQL3 % {'te1': te1, 'te2': te2}
    data.append(get_something(SPARQL1,te1,te2,1))
    data.append(get_something(SPARQL1, te2, te1,1))
    data.append(get_something(SPARQL2, te1, te2,2))
    data.append(get_something(SPARQL2, te2, te1,2))
    data.append(get_something(SPARQL3, te1, te2,3))
    data.append(get_something(SPARQL3, te2, te1,3))
    pprint(data)
create_dataset(debug = True)


#
# SPARQL1 = '''SELECT DISTINCT ?r1 ?r2 WHERE { ?uri  %(r1)s %(te1)s. ?uri %(r2)s %(te2)s . } '''
#     SPARQL2 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s %(r1)s ?uri.  %(te2)s %(r2)s ?uri . } '''
#     SPARQL3 = '''SELECT DISTINCT ?r1 ?r2 WHERE { %(te1)s %(r1)s ?uri.  ?uri %(r2)s %(te2)s . } '''
#
# {u'_id': u'6ff03a568e2e4105b491ab1c1411c1ab',
#  u'corrected_question': u'What tv series can be said to be related to the sarah jane adventure and dr who confidential?',
#  u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?uri <http://dbpedia.org/ontology/related> <http://dbpedia.org/resource/The_Sarah_Jane_Adventures> . ?uri <http://dbpedia.org/ontology/related> <http://dbpedia.org/resource/Doctor_Who_Confidential> . }',
#  u'sparql_template_id': 7,
#  u'verbalized_question': u'What is the <television show> whose <relateds> are <The Sarah Jane Adventures> and <Doctor Who Confidential>?'}

