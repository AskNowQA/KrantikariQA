'''
    Depoying Krantikari models.
'''

from __future__ import absolute_import

# Generic imports
import os
import sys
import json
import math
import pickle
import warnings
import requests
import traceback
import numpy as np
import pandas as pd
from progressbar import ProgressBar

# Keras Imports
from keras import optimizers
from keras.models import load_model
import keras.backend.tensorflow_backend as K
from keras.preprocessing.sequence import pad_sequences

# Local imports
import network as n
import data_preparation_rdf_type as drt
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils
from utils import prepare_vocab_continous as vocab_master
import one_file as of
import krantikari_new as kn

DEBUG = True
POINTWISE = False
FLAG = True
MULTI_HOP = False
EMBEDDING = "glove"
MAX_SEQ_LENGTH = 25
DATASET = of.DATASET
CORECHAIN_MODEL_DIR = './data/models/core_chain/birnn_dot/lcquad/model_05/model.h5'
RDFCHECK_MODEL_DIR = './data/models/rdf/%(data)s/model_01/model.h5' % {'data':DATASET}
RDFEXISTENCE_MODEL_DIR = './data/models/type_existence/%(data)s/model_00/model.h5' % {'data':DATASET}
INTENT_MODEL_DIR = './data/models/intent/%(data)s/model_00/model.h5' % {'data':'lcquad'}
RELATIONS_LOC = os.path.join(n.COMMON_DATA_DIR, 'relations.pickle')
RDF_TYPE_LOOKUP_LOC = 'data/data/common/rdf_type_lookup.json'


# Configure at every run!
GPU = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

# Set the seed to clamp the stochastic nature.
np.random.seed(42) # Random train/test splits stay the same between runs

# Global variables

# Global variables
dbp = db_interface.DBPedia(_verbose=True, caching=True)
vocab, vectors = vocab_master.load()

reverse_vocab = {}
for keys in vocab:
    reverse_vocab[vocab[keys]] = keys


# sparql template supported by krantikari currently.
sparql_1hop_template = of.sparql_1hop_template
sparql_boolean_template = of.sparql_boolean_template
sparql_2hop_1ent_template = of.sparql_2hop_1ent_template
sparql_2hop_2ent_template = of.sparql_2hop_2ent_template
rdf_constraint_template = of.rdf_constraint_template


# some auxilary modules
relations = of.load_relation()
non_inverse_relations = pickle.load(open(of.RELATIONS_LOC))

reverse_relations = {}
for keys in relations:
    reverse_relations[relations[keys][0]] = [keys] + relations[keys][1:]

reverse_rdf_dict = of.load_reverse_rdf_type()

with K.tf.device('/gpu:' + GPU):
    metric = of.rank_precision_metric(10)
    model_corechain = load_model(CORECHAIN_MODEL_DIR, {'custom_loss': of.custom_loss, 'metric': metric})
    model_rdf_type_check = load_model(RDFCHECK_MODEL_DIR, {'custom_loss': of.custom_loss, 'metric': metric})
    model_rdf_type_existence = load_model(RDFEXISTENCE_MODEL_DIR)
    model_question_intent = load_model(INTENT_MODEL_DIR)

'''
    Some aux functions 
'''
def return_entites(question):
    '''
        uses EARL to find all the entites present in the question.
        :param question: question in non-vectorized version
        :return: entities list
    '''
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


def relation_lookup(rel):
    global non_inverse_relations
    global relations
    try:
        return non_inverse_relations[rel][0]
    except:
        '''
            optional -> add it to the non_inverse_relation
            necessary -> add it to relations
        '''
        surface_form = dbp.get_label(rel)
        surface_form_tokenized = nlutils.tokenize(surface_form)
        surface_form_tokenized_id = embeddings_interface.vocabularize(surface_form_tokenized)
        counter = len(non_inverse_relations)
        non_inverse_relations[rel] = [counter, surface_form, surface_form_tokenized, surface_form_tokenized_id]
        # counter = counter + 1
        value = [counter, surface_form, surface_form_tokenized, surface_form_tokenized_id]
        new_key = value[0]
        value[0] = rel
        # value.append(new_key)
        relations[new_key] = value
        return non_inverse_relations[rel][0]

def rel_to_id(data):
    '''
    :param data: data which gets passed around in the relation aggregate.py file.
    :return: idfy output
    '''
    data['parsed-data'] = {}
    data['parsed-data']['path'] = [-1]
    data['parsed-data']['path_id'] = [-1]
    hop1 = []

    for r in data['uri']['hop-1-properties']:
        temp = [r[0],relation_lookup(r[1])]
        hop1.append(temp)
    hop2 = []

    for r in data['uri']['hop-2-properties']:
        temp = [ r[0], relation_lookup(r[1]) , r[2], relation_lookup(r[3]) ]
        hop2.append(temp)

    data['uri']['hop-1-properties'] = hop1
    data['uri']['hop-2-properties'] = hop2

    return data

core_chain_accuracy_counter = 0
rank_precision_runtime = 0


def return_sparql(question,entity=False):
    '''
        The main logic or the core of krantikari.

    :return: sparql,query_graph,error_code

    error_code = 'no_entity' --> No entity returned by the entity linker
                = 'no_best_path' --> No path created
                = 'entity_server_error' --> some server issue at entity linking server

    '''
    error_code = ''
    if not entity:
        try:
            entites = return_entites(question)
        except ValueError:
            return '', '', 'entity_server_error'
        try:
            entites.remove('null')
        except ValueError:
            pass
        if len(entites) >= 3:
            entites = entites[:1]
        if not entites:
            return '','','no_entity'
        '''
            Use krantikari_new file to fetch the sub-graph in the form of paths.
        '''
    else:
        entites = entity

    query_graph = {}
    query_graph['question'] = question

    question_vector = embeddings_interface.vocabularize(nlutils.tokenize(question), _embedding=EMBEDDING)
    question_vector_inverse_vocab = np.asarray([vocab[key] for key in question_vector])
    question_vector_padded = of.pad_question(question_vector_inverse_vocab)

    intent = of.question_intent(model_question_intent, question_vector_padded)
    query_graph['intent'] = intent

    if intent == 'ask':
        training_data = kn.Krantikari_v2(_question=question, _entities=entites, _model_interpreter="",
                                         _dbpedia_interface=dbp,
                                         _training=True, _ask=True, _qald=True)
    else:
        training_data = kn.Krantikari_v2(_question=question, _entities=entites, _model_interpreter="",
                                         _dbpedia_interface=dbp,
                                         _training=True, _ask=False, _qald=True)
    # final_uri_data = training_data.data
    final_uri_data = {}
    final_uri_data['uri'] = training_data.data
    '''
        >Idfy thee relation
        >Change the format of data
        >Pass it through create dataset function.
    '''
    final_uri_data = rel_to_id(final_uri_data)
    final_uri_data['parsed-data']['entity'] = entites
    question, positive_path, negative_paths, no_positive_path = of.construct_paths(final_uri_data, relations=relations,
                                                                                   qald=True)
    nps = [ne.tolist() for ne in negative_paths]
    # pp = [positive_path.tolist()]
    paths = nps

    index = of.prune_candidate_space(question, paths, len(paths))
    paths = [paths[i] for i in index]

    '''
        Converting paths to numpy array
    '''
    for i in range(len(paths)):
        paths[i] = np.asarray(paths[i])
    paths = np.asarray(paths)

    '''
        Ranking for the best core chain
    '''
    core_chain_accuracy_counter = 0
    question, paths, positive_path, negative_paths, core_chain_accuracy_counter, best_path, mrr = of.core_chain_accuracy(
        question, paths, positive_path, negative_paths, core_chain_accuracy_counter, model_corechain, no_positive_path)

    query_graph['best_path'] = best_path

    if intent == 'ask' and best_path != '':
        return 'true',query_graph,''
    elif intent == 'ask' and best_path == '':
        return 'false',query_graph,''

    if best_path != '':

        '''
            rdf type contraints.
        '''
        rdf_constraint = of.rdf_constraint_check(question_vector_padded, model_rdf_type_existence)
        query_graph['rdf_constraint'] = False if rdf_constraint == 2 else True
        query_graph['rdf_constraint_type'] = ['x', 'uri', 'none'][rdf_constraint]

        if rdf_constraint != 2:
            rdf_candidates = of.rdf_type_candidates(final_uri_data, best_path, vocab, relations, reverse_vocab,
                                                    only_x=rdf_constraint == 0,
                                                    core_chain=True)
            if rdf_candidates:
                output = of.rank_precision_runtime(model_rdf_type_check, question, rdf_candidates[0],
                                                   rdf_candidates, 180, MAX_SEQ_LENGTH, rdf=True)
                # rdf_best_path = convert_path_to_text(rdf_candidates[np.argmax(output[1:])])
                query_graph['rdf_best_path'] = rdf_candidates[np.argmax(output[1:])]
            else:
                query_graph['rdf_best_path'] = []
        query_graph['entities'] = entites
        sparql = of.query_graph_to_sparql(query_graph)
        return sparql, query_graph, error_code
    else:
        return '', '', 'no_best_path'







