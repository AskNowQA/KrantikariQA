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
    '''

    counter = 0
    cd_node = cd.create_data_node(_predicate_blacklist,_relation_file,_qald)
    successful_data = []
    unsuccessful_data = []
    for node in _dataset:
        try:
            temp = {}
            temp['node'] = node
            temp['hop1'],temp['hop2'],temp['path'],temp['entity'],temp['constraints'],flag =  cd_node.dataset_preparation_time(node)
            if flag:
                successful_data.append(temp)
            else:
                temp['error'] = 'true path not generated'
                unsuccessful_data.append(temp)
        except:
            temp = {}
            temp['node'] = node
            temp['error'] = str(traceback.print_exc())
            unsuccessful_data.append(temp)
        print ("done with, ", counter)
        counter = counter + 1

    return successful_data,unsuccessful_data