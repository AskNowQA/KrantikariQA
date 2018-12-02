'''
    Orchestrates the whole pipeline for generating the core-chain candidates.
'''

import datasetPreparation.create_dataset as cd
import traceback
import pathlib
import json
import sys
import os
# Checks if the location exists and if not create a new one.
create_dir = lambda dir_location: pathlib.Path(dir_location).mkdir(parents=True, exist_ok=True)



_save_location_success = 'data/data/raw/%(dataset)s/success'
_save_location_unsuccess = 'data/data/raw/%(dataset)s/unsuccess'

file_name = '.json'


def convert_qald_to_lcquad(dataset):
    dataset = dataset['questions']
    new_dataset = []
    for index,node in enumerate(dataset):
        d = dict(corrected_question=node['question'][0]['string'], verbalized_question=node['question'][0]['string'],
                 _id=index, sparql_query=node['query']['sparql'].replace('\n', ' '), sparql_template_id=999)
        new_dataset.append(d)
    return new_dataset


def run(_dataset,_save_location_success,_save_location_unsuccess,
        _file_name,_predicate_blacklist,_relation_file,return_data,_qald=False):
    '''

    :param dataset: a list of data node.
    :param _save_location_success: location where the data where everything is correct is stored.
    :param _save_location_unsuccess: location where the data where there was some error is stored.
    :param file_name: name f the file in which data is stored.
    :param return_data: returns the success and unsucess data if true else nothing is returned
    :return:

    Note :- flag is used for determining whether the correct path was generated in the dataset
    generation process or not.

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
                'constraint_found_in_data_generated':False
            },
            'rdf_constraint' : {}
        }
    '''


    #creating dir, if it doesn't exists
    create_dir(_save_location_success)
    create_dir(_save_location_unsuccess)
    fullpath_success = os.path.join(_save_location_success,_file_name)
    fullpath_unsuccess = os.path.join(_save_location_unsuccess,_file_name)

    counter = 0
    cd_node = cd.CreateDataNode(_predicate_blacklist=_predicate_blacklist, _relation_file=_relation_file, _qald=_qald)
    successful_data = []
    unsuccessful_data = []

    for node in _dataset:
        try:
            data =  cd_node.dataset_preparation_time(_data_node=node,rdf=True)
            data['error_flag']['aux_error'] = False
            if data['error_flag']['path_found_in_data_generated'] and \
                    data['error_flag']['constraint_found_in_data_generated']:
                successful_data.append(data)
            else:
                unsuccessful_data.append(data)
        except:
            temp = {}
            temp['node'] = node
            temp['error_flag'] = {}
            temp['error_flag']['aux_error'] = traceback.format_exc()
            unsuccessful_data.append(temp)
        print ("done with, ", counter)
        counter = counter + 1

    if _qald:
        new_unsuccessful_data = []
        for u in unsuccessful_data:
            try:
                if u['error_flag']['path_found_in_data_generated'] == False:
                    new_unsuccessful_data.append(u)
            except:
                continue

        unsuccessful_data = new_unsuccessful_data

    json.dump(successful_data,open(fullpath_success,'w+'))
    json.dump(unsuccessful_data,open(fullpath_unsuccess,'w+'))
    print("the len of successfull data is ", len(successful_data))
    print("the len of unsuccessfull data is ", len(unsuccessful_data))

    if return_data:
        return successful_data,unsuccessful_data

if __name__ == "__main__":
    start_index = sys.argv[1]
    end_index = sys.argv[2]
    dataset = sys.argv[3]




    if dataset == 'lcquad':
        _save_location = 'data/data/raw/lcquad'
    else:
        _save_location = 'data/data/raw/qald'

    file_name = start_index+file_name

    _save_location_success = _save_location_success % {'dataset':dataset}
    _save_location_unsuccess = _save_location_unsuccess % {'dataset':dataset}

    pb = open('resources/predicate.blacklist').readlines()
    pb[-1] = pb[-1] + '\n'
    pb = [r[:-1] for r in pb]




    if dataset == 'lcquad':
        _dataset = json.load(open('resources/lcquad_data_set.json'))
    elif dataset == 'qald':
        qald_train = json.load(open('resources/qald-7-train-multilingual.json'))
        _dataset = convert_qald_to_lcquad(qald_train)


    if end_index == -1:
        _dataset = _dataset[int(start_index):]
    else:
        _dataset = _dataset[int(start_index):int(end_index)]


    if dataset == 'lcquad':
        run(_dataset=_dataset, _save_location_success=_save_location_success
            , _save_location_unsuccess=_save_location_unsuccess,
            _file_name=file_name,
            _predicate_blacklist=pb, _relation_file={}, return_data=False, _qald=False)

    if dataset == 'qald':
        run(_dataset=_dataset, _save_location_success=_save_location_success
            , _save_location_unsuccess=_save_location_unsuccess,
            _file_name=file_name,
            _predicate_blacklist=pb, _relation_file={}, return_data=False, _qald=True)



