'''
    Orchestrates the whole pipeline for generating the core-chain candidates.
'''

import datasetPreparation.create_dataset as cd
import traceback

def run(_dataset,_save_location,_file_name,_predicate_blacklist,_relation_file,_qald=False):
    '''

    :param dataset: a list of data node.
    :param save_location: location where the data would be stored.
    :param file_name: name of the file in which data is stored.
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
            temp['error_flag']['aux_error'] = str(traceback.print_exc())
            unsuccessful_data.append(temp)
        print ("done with, ", counter)
        counter = counter + 1

    return successful_data,unsuccessful_data