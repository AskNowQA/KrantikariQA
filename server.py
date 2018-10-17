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

import onefile as qa

from datasetPreparation import rdf_candidates as rdfc
from configs import config_loader as cl
from datasetPreparation import entity_subgraph as es
from utils import dbpedia_interface as dbi
from utils import natural_language_utilities as nlutils
from utils.goodies import *
from utils import embeddings_interface as ei
from utils import query_graph_to_sparql as qgts
ei.__check_prepared__()

quesans, dbp, subgraph_maker, relations, parameter_dict = None, None, None, None, None


# path --> 'http://dbpedia.org/property/stadium'
vocabularize_relation = lambda path: ei.vocabularize(nlutils.tokenize(dbp.get_label(path))).tolist()
vocabularize_specia_char = lambda char: [ei.SPECIAL_CHARACTERS.index(char)]

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


def convert_path_to_id(path):
    """


    :param path: ['+',2,'-',4] where 2 and 4 refers to relation key
    :return:

    """
    idfy_path = []
    for p in path:
        if p in ['+', '-']:
            idfy_path.append(ei.SPECIAL_CHARACTERS.index(p))
        else:
            idfy_path += inverse_relations[int(p)][3].tolist()

    return idfy_path


def run():
    """
        Pulls the parameters from disk
        Instantiate QuestionAnswering class (onefile).

        Thus, all models are loaded.
        Help loading the model, then start the server.

        CHANGE MAJOR CONFIGS HERE
    :return:
    """
    global quesans, dbp, subgraph_maker, relations, parameter_dict

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
    parameter_dict['_dataset_specific_data_dir'] = qa._dataset_specific_data_dir
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
    relations = pickle.load(open(os.path.join(qa.COMMON_DATA_DIR,'relations.pickle'),'rb'),encoding='bytes')

    quesans = qa.QuestionAnswering(parameters=parameter_dict, pointwise=training_config,
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

    _graph = {
        'entities' : [],
        'best_path' : [],
        'intent' : "",
        'rdf_constraint':False,
        'rdf_constraint_type':'',
        'rdf_best_path':''
    }

    # @TODO: put in type checks if needed here.
    question_id = ei.vocabularize(nlutils.tokenize(question))

    entities = get_entities(question)
    _graph['entities'] = entities

    if not entities: raise NoEntitiesFound

    hop1, hop2 = subgraph_maker.subgraph(entities, question, {}, _use_blacklist=True, _qald=False)
    if len(hop1) == 0 and len(hop2) == 0: raise NoPathsFound



    _hop1 = [vocabularize_specia_char(h[0])+vocabularize_relation(h[1]) for h in hop1]
    _hop2 = [vocabularize_specia_char(h[0])+vocabularize_relation(h[1])+
              vocabularize_specia_char(h[2])+vocabularize_relation(h[3]) for h in hop2]
    paths = _hop1+ _hop2
    paths_sf = hop1+hop2

    if parameter_dict['corechainmodel'] == 'slotptr':
        paths_rel1_sp, paths_rel2_sp, _, _ = qa.create_rd_sp_paths(paths)
        output = quesans._predict_corechain(question_id, paths, paths_rel1_sp, paths_rel2_sp)
    elif parameter_dict['corechainmodel'] == 'bilstm_dot':
        output = quesans._predict_corechain(question_id, np.asarray(paths))
    else:
        raise BadParameters('corechainmodel')

    best_path_index = np.argmax(output)
    best_path_sf = paths_sf[best_path_index]
    # # ##############################################
    # """
    #     Intent, rdftype prediction
    #
    #     Straightforward.
    #
    #     Metrics: accuracy
    # """
    # # ##############################################
    # # Get intent
    intent_pred = np.argmax(quesans._predict_intent(question_id))
    rdftype_pred = np.argmax(quesans._predict_rdftype(question_id))
    intent = qa.INTENTS[intent_pred]
    rdftype = qa.RDFTYPES[rdftype_pred]

    _graph['best_path'] = best_path_sf
    _graph['intent'] = intent



    if rdftype != 'none':
        _graph['rdf_constraint'] = True
        x_constraint,uri_constraint = rdfc.generate_rdf_candidates(path=best_path_sf,topic_entity=entities,dbp=dbp)
        constraint_sf = x_constraint+uri_constraint
        x_constraint = [ei.vocabularize(nlutils.tokenize(" ".join(['x',dbp.get_label(x)]))) for x in x_constraint]
        uri_constraint = [ei.vocabularize(nlutils.tokenize(" ".join(['uri',dbp.get_label(uri)]))) for uri in uri_constraint]
        constraint = x_constraint+uri_constraint
        constraint_output_index = np.argmax(quesans._predict_rdfclass(question_id,constraint))
        constraint_output_sf = constraint_sf[constraint_output_index]
        if constraint[constraint_output_index][0] == 'x':
            _graph['rdf_constraint_type'] = 'x'
            _graph['rdf_best_path'] = constraint_output_sf
        else:
            _graph['rdf_constraint_type'] = 'uri'
            _graph['rdf_best_path'] = constraint_output_sf


    sparql = qgts.convert_runtime(_graph)


    return intent,rdftype,best_path_sf,sparql




run()
question = 'Who is the president of America ?'
answer_question(question)