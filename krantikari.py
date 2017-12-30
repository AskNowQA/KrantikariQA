"""
    Author: geraltofrivia; saist1993
    The objective of this file to implement Krantikari QA.

    NOTE: A lot of code scavenged from preProcessing.py, some from parser.py


    Needs:
        - trained Keras model
        - working dbpedia interface
        - question
        - topic entity

    Pseudocode:
            +------------------------------------------------------------------------+
            |                  get a question                                        |
            |                       +                                                |
            |                       |                                                |
            |                       v                                                |
            |                  get entities                                          |
            |                   +         +                                          |
            |                   |         |                                          |
            |      1 entity  <--+         +-->   2 entities                          |
            |            +                              +                            |
            |            |                              |                            |
            |            v                              v                            |
            |      get subgraph                   get subgraph for  - S_e1           |
            |            +                            first entity                   |
            |            |                                  +                        |
            |            v                                  |                        |
            |    generate 1-hop paths                       v                        |
            |       ENT1 +/- R1                   get subgraph for  - S_e1           |
            |            |                           second entity                   |
            |            v                                  +                        |
            |     Pick top k options                        |                        |
            |            +                                  v                        |
            |            |                     Intersect them - S = S_e1 AND S_e2    |
            |            v                                  +                        |
            |    generate 2-hop paths                       |                        |
            |     ENT1 +/- R1 +/- R2                        v                        |
            |            |                       Generate paths in S like:           |
            |            v                       E1 +- R1 +- R2 +- E2                |
            |     Rank, pick top k                          +                        |
            |            +                                  |                        |
            |            |                                  v                        |
            |            |                         Rank, pick top k                  |
            |            |                                  +                        |
            |            |                                  |                        |
            |            |                                  |                        |
            |            |                                  |                        |
            |            +->  post ranking filters   <------+                        |
            |                        +                                               |
            |                        |                                               |
            |                        v                                               |
            |             convert core chain to sparql                               |
            |                        +                                               |
            |                        |                                               |
            |                        v                                               |
            |       detect if RDF:type constraints are needed                        |
            |                        +                                               |
            |                        |                                               |
            |                        v                                               |
            |                      ANSWER                                            |
            |                                                                        |
            +------------------------------------------------------------------------+


    Current Version: just one topic entity, multiple hops @TODO: Keep updating this.
"""

# Imports
import os
import json
import random
import pickle
import warnings
import traceback
import numpy as np
from pprint import pprint

# Local file imports
from utils import model_interpreter
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils

# Some MACROS
DEBUG = True
K_1HOP_GLOVE = 20
K_1HOP_MODEL = 5
K_2HOP_GLOVE = 10
K_2HOP_MODEL = 5
MODEL_DIR = 'data/training/multi_path_mini/model_00/model.h5'

# Global Variables
dbp = db_interface.DBPedia(_verbose=True, caching=False)  # Summon a DBpedia interface
model = model_interpreter.ModelInterpreter()  # Model interpreter to be used for ranking


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def convert_core_chain_to_sparql(_core_chain):  # @TODO
    pass


def get_hop2_subgraph(_entity, _predicate, _right=True):
    """
        Function fetches the 2hop subgraph around this entity, and following this predicate
    :param _entity: str: URI of the entity around which we need the subgraph
    :param _predicate: str: URI of the predicate with which we curtail the 2hop graph (and get a tangible number of ops)
    :param _right: Boolean: True -> _predicate is on the right of entity (outgoing), else left (incoming)

    :return: List, List: ight (outgoing), left (incoming) preds
    """
    # Get the entities which will come out of _entity +/- _predicate chain
    intermediate_entities = dbp.get_entity(_entity, [_predicate], _right)

    # Filter out the literals, and keep uniques.
    intermediate_entities = list(set([x for x in intermediate_entities if x.startswith('http://dbpedia.org/resource')]))

    left_predicates, right_predicates = [], []  # Places to store data.

    for entity in intermediate_entities:
        temp_l, temp_r = dbp.get_properties(_uri=entity, label=False)
        left_predicates += temp_l
        right_predicates += temp_r

    return list(set(right_predicates)), list(set(left_predicates))


def similar_predicates(_question, _predicates, _return_indices=False, _k=5):
    """
        Function used to tokenize the question and compare the tokens with the predicates.
        Then their top k are selected.

        @TODO: Verify this chain of thought.

    :param _question: a string of question
    :param _predicates: a list of strings (surface forms/uri? @TODO: Verify)
    :param _return_indices: BOOLEAN to only return indices or not.
    :param _k: int :- select top k from sorted list
    :return: a list of strings (subset of predicates)
    """
    # Tokenize question
    qt = nlutils.tokenize(_question, _remove_stopwords=False)

    # Vectorize question
    v_qt = embeddings_interface.vectorize(qt)

    # Vectorize predicates
    v_pt = np.asarray([np.mean(embeddings_interface.vectorize(nlutils.tokenize(x)), axis=0) for x in _predicates])

    # Compute similarity
    similarity_mat = np.dot(v_pt, np.transpose(v_qt))

    # Find the best scoring values for every path
    # Sort ( best match score for each predicate) in descending order, and choose top k
    argmaxes = np.argsort(np.max(similarity_mat, axis=1))[::-1][:_k]

    if _return_indices:
        return argmaxes

    # Use this to choose from _predicates and return
    return [_predicates[i] for i in argmaxes]


def runtime(_question, _entities, _return_core_chains=False, _return_answers=False):
    """
        This function inputs one question, and topic entities, and returns a SPARQL query (or s'thing else)

    :param _question: a string of question
    :param _entities: a list of strings (each being a URI)
    :return: SPARQL/CoreChain/Answers (and or)
    """

    # Vectorize the question
    v_q = embeddings_interface.vectorize(nlutils.tokenize(_question))

    # Algo differs based on whether there's one topic entity or two
    if len(_entities) == 1:

        # Get 1-hop subgraph around the entity
        right_properties, left_properties = dbp.get_properties(_uri=_entities[0], label=False)

        # @TODO: Use predicate whitelist/blacklist to trim this shit.

        # Get the surface forms of Entity and the predicates
        entity_sf = dbp.get_label(_resource_uri=_entities[0])
        right_properties_sf = [dbp.get_label(x) for x in right_properties]
        left_properties_sf = [dbp.get_label(x) for x in left_properties]

        # WORD-EMBEDDING FILTERING
        right_properties_filter_indices = similar_predicates(_question=_question,
                                                             _predicates=right_properties_sf,
                                                             _return_indices=True,
                                                             _k=K_1HOP_GLOVE)
        left_properties_filter_indices = similar_predicates(_question=_question,
                                                            _predicates=left_properties_sf,
                                                            _return_indices=True,
                                                            _k=K_1HOP_GLOVE)

        # Impose these indices to generate filtered predicate list.
        right_properties_filtered = [right_properties_sf[i] for i in right_properties_filter_indices]
        left_properties_filtered = [left_properties_sf[i] for i in left_properties_filter_indices]

        # Generate 1-hop paths out of them
        paths_sf = [nlutils.tokenize(entity_sf) + ['+'] + nlutils.tokenize(_p) for _p in right_properties_filtered]
        paths_sf += [nlutils.tokenize(entity_sf) + ['-'] + nlutils.tokenize(_p) for _p in left_properties_filtered]

        # Vectorize these paths.
        v_ps = [embeddings_interface.vectorize(path) for path in paths_sf]

        # MODEL FILTERING
        hop1_indices, hop1_scores = model.rank(_v_q=v_q,
                                               _v_ps=v_ps,
                                               _return_only_indices=False,
                                               _k=K_1HOP_MODEL)

        # Impose indices on the paths.
        ranked_paths_hop1 = [paths_sf[i] for i in hop1_indices]

        if DEBUG:
            pprint(ranked_paths_hop1)

        # Collect URI of predicates so filtered (for 2nd hop)
        left_properties_filtered, right_properties_filtered = [], []

        # Gather all the left and right predicates (from paths selected by the model)
        for i in hop1_indices:

            hop1_path = paths_sf[i]

            # See if it is from the left or right predicate set.
            if '-' in hop1_path:
                # This belongs to the left pred list.
                # Offset index to match to left_properties_filter_indices index.

                i -= len(right_properties_filtered)
                predicate = left_properties[left_properties_filter_indices[i]]
                left_properties_filtered.append(predicate)

            else:
                # This belongs to the right pred list.
                # No offset needed

                predicate = right_properties[right_properties_filter_indices[i]]
                right_properties_filtered.append(predicate)

        if DEBUG:
            print("Check URI of filtered paths. Left, then right.")
            pprint(left_properties_filtered)
            pprint(right_properties_filtered)

        """
            2 - Hop COMMENCES

            Note: Switching to LC-QuAD nomenclature hereon. Refer to /resources/nomenclature.png
        """
        e_in_in_to_e_in = []
        e_in_to_e_in_out = []
        e_out_to_e_out_out = []
        e_out_in_to_e_out = []

        for pred in right_properties_filtered:
            temp_r, temp_l = get_hop2_subgraph(_entity=_entities[0], _predicate=pred, _right=True)
            e_out_to_e_out_out += temp_r
            e_out_in_to_e_out += temp_l

        for pred in left_properties_filtered:
            temp_r, temp_l =  get_hop2_subgraph(_entity=_entities[0], _predicate=pred, _right=False)
            e_in_to_e_in_out += temp_r
            e_in_in_to_e_in += temp_l

        # Get uniques
        e_in_in_to_e_in = list(set(e_in_in_to_e_in))
        e_in_to_e_in_out = list(set(e_in_to_e_in_out))
        e_out_to_e_out_out = list(set(e_out_to_e_out_out))
        e_out_in_to_e_out = list(set(e_out_in_to_e_out))

        # Predicates generated. @TODO: Use predicate whitelist/blacklist to trim this shit.

        if DEBUG:
            print("HOP2 Subgraph")
            pprint(e_in_in_to_e_in)
            pprint(e_in_to_e_in_out)
            pprint(e_out_to_e_out_out)
            pprint(e_out_in_to_e_out)

        # Get their surface forms
        e_in_in_to_e_in_sf = [dbp.get_label(x) for x in e_in_in_to_e_in]
        e_in_to_e_in_out_sf = [dbp.get_label(x) for x in e_in_to_e_in_out]
        e_out_to_e_out_out_sf = [dbp.get_label(x) for x in e_out_to_e_out_out]
        e_out_in_to_e_out_sf = [dbp.get_label(x) for x in e_out_in_to_e_out]

        # WORD-EMBEDDING FILTERING
        e_in_in_to_e_in_filter_indices = similar_predicates(_question = _question,
                                                            _predicates=e_in_in_to_e_in_sf,
                                                            _return_indices=True,
                                                            _k = K_2HOP_GLOVE)
        e_in_to_e_in_out_filter_indices = similar_predicates(_question = _question,
                                                             _predicates=e_in_to_e_in_out_sf,
                                                             _return_indices=True,
                                                             _k = K_2HOP_GLOVE)
        e_out_to_e_out_out_filter_indices = similar_predicates(_question = _question,
                                                               _predicates=e_out_to_e_out_out_sf,
                                                               _return_indices=True,
                                                               _k = K_2HOP_GLOVE)
        e_out_in_to_e_out_filter_indices = similar_predicates(_question=_question,
                                                              _predicates=e_out_in_to_e_out_sf,
                                                              _return_indices=True,
                                                              _k=K_2HOP_GLOVE)

        # Impose these indices to generate filtered predicate list.
        e_in_in_to_e_in_filtered = [e_in_in_to_e_in_sf[i] for i in e_in_in_to_e_in_filter_indices]
        e_in_to_e_in_out_filtered = [e_in_to_e_in_out_sf[i] for i in e_in_to_e_in_out_filter_indices]
        e_out_to_e_out_out_filtered = [e_out_to_e_out_out_sf[i] for i in e_out_to_e_out_out_filter_indices]
        e_out_in_to_e_out_filtered = [e_out_in_to_e_out_sf[i] for i in e_out_in_to_e_out_filter_indices]

        # Generate 2-hop paths out of them.

        # Vectorize these paths

        # MODEL FILTERING

        # Impose indices




    if len(_entities) >= 2:
        pass


if __name__ == "__main__":
    """
        TEST1 : Accuracy of similar_predicates
    """
    q = 'Who is the president of United States'
    p = ['abstract', 'motto', 'population total', 'official language', 'legislature', 'lower house', 'president',
         'leader', 'prime minister']
    e = 'http://dbpedia.org/resource/United_States'
    # print(similar_predicates(q, p, 5))
    print runtime(q, [e])
