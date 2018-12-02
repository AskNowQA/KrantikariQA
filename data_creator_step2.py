'''

		>write a script to collect everything from a folder
		>Use the same script to find all relations
		>Create a piclke of relation in that specific form
		>Idfy everything with respect to relation
		>Store it

        The relation file structure would be
            [
                'http://dbpedia.org/property/ratifiers' : ['ID','SF','SF Tokenized','SF ID']
            ]
'''
import pickle
import json
import os
import numpy as np

from utils import dbpedia_interface as dbi
from utils import natural_language_utilities as nlutils
from utils import embeddings_interface as ei

ei.__check_prepared__()

def collect_files(dir_location):
    '''

    :param dir_location: json file location (No name needed) --> /data/data/raw/lcquad/success
    :return: big json combining all the files in the given location

    '''

    file_list = [os.path.join(dir_location,file) for file in os.listdir(dir_location)]
    json_list = [json.load(open(file)) for file in file_list]
    final_data = []
    for node in json_list:
        final_data = final_data + node

    return final_data

def update_relation_dict(relation,relation_dict,dbp):
    '''

    Updates the relation dict if the relation doesn't exists
        Also returns the id for the same

    :param relation: 'http://dbpedia.org/property/services'
    :param relation_dict: {}
    :return: id version of the relation and as well as relation_dict

    ['ID','SF','SF Tokenized','SF ID']

    'SF ID' = embeddings_interface.vocabularize(surface_form_tokenized)

    '''
    if relation in relation_dict.keys():
        rel_id = relation_dict[relation][0]
    else:
        rel_id = len(relation_dict)
        surface_form = dbp.get_label(relation)
        surface_form_tokenized = nlutils.tokenize(surface_form)
        relation_dict[relation] = [len(relation_dict),surface_form,surface_form_tokenized
            ,ei.vocabularize_idspace(surface_form_tokenized)]

    return rel_id,relation_dict

def idfy_path(path,relation_dict,dbp):
    '''

    :param path: ['+', 'http://dbpedia.org/property/services', '-','optinalpath']
    :return: checks if the relations in path exists in dict and if not update the relation file
        > Also return an id version of the path.


    '''

    if len(path) == 2:
        rel_id,relation_dict = update_relation_dict(relation=path[1],relation_dict=relation_dict, dbp=dbp)
        return [path[0],rel_id],relation_dict
    else:
        rel1_id,relation_dict = update_relation_dict(relation=path[1],relation_dict=relation_dict, dbp=dbp)
        rel2_id,relation_dict = update_relation_dict(relation=path[3],relation_dict=relation_dict, dbp=dbp)
        return [path[0],rel1_id,path[2],rel2_id],relation_dict

def idfy_const(const,relation_dict,dbp):
    '''

    :param const: 'http://dbpedia.org/property/services' constraint has no sign
    :param relation_dict:
    :return: idfy const,updated relation dict
    '''
    return update_relation_dict(const,relation_dict,dbp)


def idfy_relations_in_node(node,relation_dict,dbp):
    '''

    Given a node, idfy all the relation and if the relation doesn't exists in the relation
    dictionary, update the rel dict.

    :param node: data node
    :param relation_dict:

    :return:
    '''
    if node['path']:
        node['path'],relation_dict = idfy_path(node['path'],relation_dict,dbp)

    for index,path in enumerate(node['hop1']):
        node['hop1'][index],relation_dict = idfy_path(path,relation_dict,dbp)

    for index,path in enumerate(node['hop2']):
        node['hop2'][index], relation_dict = idfy_path(path, relation_dict,dbp)

    for index,path in enumerate(node['rdf_constraint']['candidates']['uri']):
        node['rdf_constraint']['candidates']['uri'][index],relation_dict = idfy_const(path,relation_dict,dbp)

    for index,path in enumerate(node['rdf_constraint']['candidates']['x']):
        node['rdf_constraint']['candidates']['x'][index],relation_dict = idfy_const(path,relation_dict,dbp)


    return node,relation_dict

def sort_list1_wrt_list2(list1,list2):
    idlist = {v['_id']: index for index, v in enumerate(list2)}

    def getKey(node):
        return idlist[node['node']['_id']]

    return sorted(list1,key=getKey)


def vectorize_entity(entity,dbp):
    '''

    :param entity: [e1,e2] where e is of the form 'http://dbpedia.org/resource/Bill_Finger'
    :param dbp:
    :return:
    '''
    vector_ent = ei.vectorize(nlutils.tokenize(dbp.get_label(entity[0])))
    if len(entity) > 1:
        for e in entity:
            vector_ent = np.vstack((vector_ent , ei.vectorize(nlutils.tokenize(dbp.get_label(e)))))

    return vector_ent


def run(dataset):

    dataset = dataset
    _save_location_success = 'data/data/raw/%(dataset)s/success'
    _save_location_unsuccess = 'data/data/raw/%(dataset)s/unsuccess'
    relation_dict_location = 'data/data/common/relations.pickle'
    relation_dict_dir = 'data/data/common/'
    final_data_location = 'data/data/%(dataset)s/id_big_data.json'
    final_data_location_combine = 'data/data/raw/%(dataset)s/combine'
    final_data_dir = 'data/data/%(dataset)s/'

    dbp = dbi.DBPedia(caching=True)

    '''
        check if the relation dict exist and 
            if it does
                load it from disk
            else
                create a new one 
    '''

    if os.path.isfile(relation_dict_location):
        relation_dict = pickle.load(open(relation_dict_location, 'rb'))
        # To dump -> pickle.dump(relation_dict,open('data/data/common/text.pickle','wb+'))
    else:
        nlutils.create_dir(relation_dict_dir)
        relation_dict = {}
    combined_data = collect_files(_save_location_success % {'dataset':dataset})
    combined_data_un = collect_files(_save_location_unsuccess % {'dataset':dataset})

    nlutils.create_dir(final_data_location_combine % {'dataset':dataset})
    json.dump(combined_data,open(os.path.join(final_data_location_combine % {'dataset':dataset},'success.json'),'w+'))
    json.dump(combined_data_un,open(os.path.join(final_data_location_combine % {'dataset':dataset},'unsuccess.json'),'w+'))


    final_combine_data = combined_data+combined_data_un
    if dataset == 'lcquad':
        final_combine_data = sort_list1_wrt_list2(final_combine_data,json.load(open('resources/lcquad_data_set.json')))

    '''
        The final_combine_data needs to be re-ordered so that it could be directly split into training-validation-testing
    '''


    #Now here one can create a vocabulary

    for index,node in enumerate(final_combine_data):
        final_combine_data[index],relation_dict = idfy_relations_in_node(node,relation_dict=relation_dict,dbp=dbp)



    '''
    For hiearchial relation detection module one need all the relation (uri)
    have a randomly init vectors. 
    '''

    # keys = list(relation_dict.keys())
    # ei.update_vocab(keys)
    for rel in relation_dict:
        relation_dict[rel].append(ei.vocabularize_idspace([rel],False))

    print("done dumping relation")
    pickle.dump(relation_dict,open(relation_dict_location,'wb+'))

    '''
        Consider dumping here. So that alsong with relationid file and this dump
        one can do their own form of pre-processing
    '''

    #Vocabularize everything and then padding.
    '''location
        Things to vocabularize
            >question
            >path
            >hop1
            >hop2
    '''

    id_data = []


    x_id = int(ei.vocabularize_idspace(['x'])[0])
    uri_id = int(ei.vocabularize_idspace(['uri'])[0])

    for index,node in enumerate(final_combine_data):
        temp = {
            'uri' : {
                'question-id' : [int(id) for id in list(ei.vocabularize_idspace(nlutils.tokenize(node['node']['corrected_question'])))],
                'hop-2-properties' : node['hop2'],
                'hop-1-properties' : node['hop1'],
                # 'entity-id':vectorize_entity(node['entity'],dbp)
            },
            'parsed-data' : {
                'node':node['node'],
                'parsed_sparql':node['parsed_sparql'],
                'path':node['path'],
                'entity':node['entity'],
                'constraints':node['constraints'],
                'updated_sparql':node['updated_sparql'],
                'error_flag':node['error_flag']
            },
        'rdf-type-constraints' : []
        }

        rdf_candidates = []
        for candidate_id in node['rdf_constraint']['candidates']['uri']:
            rdf_candidates.append([uri_id,candidate_id])
        for candidate_id in node['rdf_constraint']['candidates']['x']:
            rdf_candidates.append([x_id, candidate_id])
        temp['rdf-type-constraints'] = rdf_candidates
        id_data.append(temp)

    #embedding interface update vocab here
    nlutils.create_dir(final_data_dir %{'dataset':dataset})
    json.dump(id_data,open(final_data_location %{'dataset':dataset},'w+'))

if __name__ == '__main__':
    run('lcquad')
    run('qald')

#update the vector file and the vocab file
#vocab file is word,index and the vector file is just vectors
ei.align_id_space()
ei.__check_prepared__()




