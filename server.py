import os
import sys
import json
import math
import torch
import pickle
import warnings
import requests
import traceback
import numpy as np

import onefile
import auxiliary as aux
import data_creator_step2 as dc2
from configs import config_loader as cl
from datasetPreparation import entity_subgraph as es
from utils import dbpedia_interface as dbi
from utils import embeddings_interface as ei
ei.__check_prepared__()

qa, dbp, subgraph_maker, relations = None, None, None, None


def get_entities(question):
    """
        uses EARL to find all the entites present in the question.
        :param question: question in non-vectorized version
        :return: entities list
    """
    headers = {
        'Content-Type': 'application/json',
    }

    # data = {"nlquery":question}
    # data = str(data)
    data = '{"nlquery":"%(p)s"}'% {"p":question}
    response = requests.post('http://sda.tech/earl/api/processQuery', headers=headers, data=data)
    a = json.loads(response.content)
    entity_list = []
    for i in range(len(a['ertypes'])):
        if a['ertypes'][i] == 'entity':
            entity_list.append(a['rerankedlists'][str(i)][0][1])
    return entity_list


def run():
    """
        Pulls the parameters from disk
        Instantiate QuestionAnswering class (onefile).

        Thus, all models are loaded.
        Help loading the model, then start the server.

        CHANGE MAJOR CONFIGS HERE
    :return:
    """
    global qa, dbp, subgraph_maker, relations

    device =  torch.device("cpu")
    dbp = dbi.DBPedia(caching=False)

    training_config = False
    training_model = "bilstm_dot"

    predicate_blacklist = open('resources/predicate.blacklist').readlines()
    predicate_blacklist[-1] = predicate_blacklist[-1] + '\n'
    predicate_blacklist = [r[:-1] for r in predicate_blacklist]
    subgraph_maker = es.CreateSubgraph(dbp, predicate_blacklist, {}, qald=False)

    # Preparing configs
    parameter_dict = cl.runtime_parameters(dataset='lcquad', training_model=training_model,
                                           training_config=training_config, config_file='configs/macros.cfg')
    parameter_dict['_dataset_specific_data_dir'] = onefile._dataset_specific_data_dir
    parameter_dict['_model_dir'] = './data/models/'

    parameter_dict['corechainmodel'] = training_model
    parameter_dict['corechainmodelnumber'] = '42'

    parameter_dict['intentmodel'] = 'bilstm_dense'
    parameter_dict['intentmodelnumber'] = '14'

    parameter_dict['rdftypemodel'] = 'bilstm_dense'
    parameter_dict['rdftypemodelnumber'] = '10'

    parameter_dict['rdfclassmodel'] = 'bilstm_dot'
    parameter_dict['rdfclassmodelnumber'] = '14'

    parameter_dict['vectors'] = ei.vectors

    # Load relations dict
    relations = pickle.load(open(os.path.join(onefile.COMMON_DATA_DIR,'relations.pickle'),'rb'),encoding='bytes')

    qa = onefile.QuestionAnswering(parameters=parameter_dict, pointwise=training_config,
                                   word_to_id=None, device=device, debug=False)



def answer_question(question):
    """
        Will get entities for the question.
        Generate subgraph
        Run corechain and aux predictions


    :param question: str: intended question to answer
    :return:
    """
    global relations, dbp

    # @TODO: put in type checks if needed here.

    entities = get_entities(question)
    hop1, hop2 = subgraph_maker.subgraph(entities, question, {}, _use_blacklist=True, _qald=False)

    _hop1 = []
    for hop in hop1:
        hop_id, relations = dc2.idfy_path(hop, relations, dbp)
        _hop1.append(hop_id)
    del hop1

    _hop2 = []
    for hop in hop2:
        hop_id, relations = dc2.idfy_path(hop, relations, dbp)
        _hop2.append(hop_id)
    del hop2

    return _hop1, _hop2


run()
question = 'Who is the chancellor of Germany ?'