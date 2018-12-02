# Takes the qald test big data and converts it into a format which
# can then be used by one file easily.

#Read the qald file.
#Find out all the relationships and update the relationship file
#idfy all the required paths and store it in a specific json format


'''

    data['uri']['question-id']
    data['uri']['hop-2-properties'] + data['uri']['hop-1-properties']
    data['parsed-data']['path']

'''

import json
import pickle
import data_creator_step2 as dc2
from utils import dbpedia_interface as dbi
from utils import embeddings_interface as ei
from utils import natural_language_utilities as nlutils


ei.__check_prepared__()
dbp = dbi.DBPedia(caching=True)
data = json.load(open('data/data/qald/qald_big_data_test.json'))

def retrive_relations(node):
    relation_list = []
    path = node['parsed-data']['path']
    print("**path is ", path)
    if path != [-1]:
        # path = [+abc,-pqr]
        p = [p[1:] for p in path]
        relation_list = relation_list + p

    hop1 = [h[-1] for h in node['uri']['hop-1-properties']]
    relation_list = relation_list + hop1
    for h in node['uri']['hop-2-properties']:
        relation_list.append(h[1])
        relation_list.append(h[-1])
    return list(set(relation_list))


def idfy_path(paths,relation_dict,dbp):
    for index,path in enumerate(paths):
        paths[index],relation_dict = dc2.idfy_path(path,relation_dict,dbp)

    return paths,relation_dict

rel = []
for node in data:
    r = retrive_relations(node)
    rel = rel + r

rel = list(set(rel))
relation_dict = pickle.load(open('data/data/common/relations.pickle', 'rb'))
for r in rel:
    _,relation_dict = dc2.update_relation_dict(relation=r,relation_dict=relation_dict,dbp=dbp)

new_data = []
for node in data:

    hop1properties,relation_dict = idfy_path(paths=node['uri']['hop-1-properties'],relation_dict=relation_dict,dbp=dbp)
    hop2properties,relation_dict = idfy_path(paths=node['uri']['hop-2-properties'],relation_dict=relation_dict,dbp=dbp)
    path = node['parsed-data']['path'][0]
    if path != -1:
        # path = [+abc,-pqr]
        p,relation_dict = dc2.idfy_const(path[1:],relation_dict,dbp)
        path = [path[0],p]

    data_s = {
        'uri':{
            'hop-1-properties' : hop1properties,
            'hop-2-properties' : hop2properties,
            'question-id' : [int(id) for id in list(ei.vocabularize_idspace(nlutils.tokenize(node['parsed-data']['corrected_question'])))]
        },
        'parsed-data':{
            'path' : path,
            'entity' : node['parsed-data']['entity'],
            'sparql_query' : node['parsed-data']['sparql_query'],
            'constraints' : node['parsed-data']['constraints'],
            'corrected_question' : node['parsed-data']['corrected_question'],
            'node': node['parsed-data']
        }
    }

    new_data.append(data_s)

#Save the relation dict

pickle.dump(relation_dict,open('data/data/common/relations.pickle', 'wb+'))
json.dump(new_data,open('data/data/qald/qald_id_big_data_test.json','w+'))








