'''
    The assumption is that the constraint is either on
    uri or x . The constraints can be count or type.
'''

import traceback
import json
from pprint import pprint

# Custom files
import utils.dbpedia_interface as db_interface

file_directory = "resources/data_set.json"
json_data = open(file_directory).read()
data = json.loads(json_data)

dbp = db_interface.DBPedia(_verbose=True, caching=False)

relations_stop_word = []

'''
    MACROS
'''

K_HOP_1 = 5 #selects the number of relations in the first hop in the right direction
K_HOP_2 = 5 #selects the number of relations in second hop in the right direction
K_HOP_1_u = 2 #selects the number of relations in second hop in the wrong direction
K_HOP_2_u = 2 #selects the number of relations in second hop in the wrong direction
PASSED = True

def get_rank_rel(_relationsip_list, rel):
    '''
        The objective is to rank the relationship using some trivial similarity measure wrt rel
    '''
    _relationsip_list
    return _relationsip_list


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


def updated_get_relationship_hop(_entity, _relation):
    '''

        This function gives all the relations after the _relationship chain. The
        difference in this and the get_relationship_hop is that it gives all the relationships from _relations[:-1],
    '''
    entities = [_entity]
    for rel in _relation:
        outgoing = rel[1]
        if outgoing:
            ''' get the objects '''
            temp = [dbp.get_entity(_entity,rel[0],outgoing=True) for ent in entities]
            entities = list(set([item for sublist in temp for item in sublist]))
        else:
            '''get the subjects '''
            temp = [dbp.get_entity(_entity, rel[0], outgoing=False) for ent in entities]
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
        rel = dbp.get_properties(ent)
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
        temp[rel] = get_set_list(get_top_k(get_rank_rel(updated_get_relationship_hop(_entity, [(rel, True)]),(rel,True)),(rel,True),hop=2))
        # temp[rel] = get_set_list(get_top_k(get_rank_rel(updated_get_relationship_hop(_entity,(rel,True)),(rel,True),hop=2)))
        outgoing_relationships.append(temp)

    for rel in rel_list[1]:
        temp = {}
        temp[rel] = get_set_list(get_top_k(get_rank_rel(updated_get_relationship_hop(_entity, [(rel, False)]),(rel,False)),(rel,False),hop=2))
        incoming_relationships.append(temp)
    return [outgoing_relationships,incoming_relationships]



final_data = []

for node in data:
    '''
        For now focusing on just simple question
    '''
    if node[u"sparql_template_id"] == 1 and not PASSED:
        '''
        {u'_id': u'9a7523469c8c45b58ec65ed56af6e306',
            u'corrected_question': u'What are the schools whose city is Reading, Berkshire?',
            u'sparql_query': u' SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/ontology/city> <http://dbpedia.org/resource/Reading,_Berkshire> } ',
             u'sparql_template_id': 1,
             u'verbalized_question': u'What are the <schools> whose <city> is <Reading, Berkshire>?'}

        '''
        data_node = node
        triples = get_triples(node[u'sparql_query'])
        data_node[u'entity'] = []
        data_node[u'entity'].append(triples[0].split(" ")[2][1:-1])
        data_node[u'training'] = {}
        data_node[u'training'][data_node[u'entity'][0]] = {}
        data_node[u'training'][data_node[u'entity'][0]][u'rel1'] = [list(set(rel)) for rel in list(dbp.get_properties(data_node[u'entity'][0]))]
        data_node[u'path'] = ["-" + triples[0].split(" ")[1][1:-1]]
        data_node[u'constraints'] = {}
        final_data.append(data_node)
        # pprint(node)
        # raw_input()
    elif node[u"sparql_template_id"] == 2 and not PASSED:
        '''
            {	u'_id': u'8216e5b6033a407191548689994aa32e',
                u'corrected_question': u'Name the municipality of Roberto Clemente Bridge ?',
                u'sparql_query': u' SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Roberto_Clemente_Bridge> <http://dbpedia.org/ontology/municipality> ?uri } ',
                u'sparql_template_id': 2,
                u'verbalized_question': u'What is the <municipality> of Roberto Clemente Bridge ?'
            }
        '''
        data_node = node
        triples = get_triples(node[u'sparql_query'])
        data_node[u'entity'] = []
        data_node[u'entity'].append(triples[0].split(" ")[0][1:-1])
        data_node[u'training'] = {}
        data_node[u'training'][data_node[u'entity'][0]] = {}
        data_node[u'training'][data_node[u'entity'][0]][u'rel1'] =  [list(set(rel)) for rel in list(dbp.get_properties(data_node[u'entity'][0]))]
        data_node[u'path'] = ["+" + triples[0].split(" ")[1][1:-1]]
        data_node[u'constraints'] = {}
        final_data.append(data_node)
        # pprint(data_node)
        # raw_input()
    elif node[u"sparql_template_id"]  == 3 or node[u"sparql_template_id"]  == 303 and not PASSED:
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
        data_node[u'training'][data_node[u'entity'][0]][u'rel1'] = [list(set(rel)) for rel in list(dbp.get_properties(data_node[u'entity'][0]))]
        data_node[u'training'][data_node[u'entity'][0]][u'rel2'] = get_stochastic_relationship_hop(data_node[u'entity'][0],[(rel1,True),(rel2,True)])
        if node[u"sparql_template_id"] == 303:
            data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[1][1:-1]}
        else:
            data_node[u'constraints'] = {}
        final_data.append(data_node)
        # pprint(data_node)
        # raw_input()

    elif node[u"sparql_template_id"]  == 5 and not PASSED:
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
        pprint(node)
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
                                                                    list(dbp.get_properties(data_node[u'entity'][0]))]
        data_node[u'training'][data_node[u'entity'][0]][u'rel2'] = get_stochastic_relationship_hop(data_node[u'entity'][0], [(rel1, False), (rel2, True)])
        if node[u"sparql_template_id"] == 305:
            data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[1][1:-1]}
        else:
            data_node[u'constraints'] = {}
        pprint(data_node)
        raw_input()
        final_data.append(data_node)

    elif node[u'sparql_template_id']  == 6:
        '''
            {
                u'_id': u'd3695db03a5e45ae8906a2527508e7c5',
                u'corrected_question': u'Who have done their PhDs under a National Medal of Science winner?',
                u'sparql_query': u'SELECT DISTINCT ?uri WHERE { ?x <http://dbpedia.org/property/prizes> <http://dbpedia.org/resource/National_Medal_of_Science> . ?uri <http://dbpedia.org/property/doctoralAdvisor> ?x  . }',
                u'sparql_template_id': 6,
                u'verbalized_question': u"What are the <scientists> whose <advisor>'s <prizes> is <National Medal of Science>?"
            }
        '''
        pprint(node)
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
                                                                    list(dbp.get_properties(data_node[u'entity'][0]))]
        data_node[u'training'][data_node[u'entity'][0]][u'rel2'] = get_stochastic_relationship_hop(
            data_node[u'entity'][0], [(rel1, False), (rel2, False)])
        if node[u"sparql_template_id"] == 305:
            data_node[u'constraints'] = {triples[2].split(" ")[0]: triples[2].split(" ")[1][1:-1]}
        else:
            data_node[u'constraints'] = {}
        pprint(data_node)
        raw_input()
        final_data.append(data_node)