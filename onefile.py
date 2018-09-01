from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import  DataLoader


from utils import prepare_vocab_continous as vocab_master
from utils import query_graph_to_sparql as sparql_constructor
from utils import dbpedia_interface as db_interface
from utils import embeddings_interface
from configs import config_loader as cl
import network_rdftype as net_rdftype
import network_intent as net_intent
import data_loader as dl
import auxiliary as aux
import network as net

import os
import sys
import json
import time
import pickle
import numpy as np
from pprint import pprint
from progressbar import ProgressBar

if sys.version_info[0] == 3: import configparser as ConfigParser
else: import ConfigParser

device = torch.device("cuda")
sparql_constructor.init(embeddings_interface)
dbp = db_interface.DBPedia(_verbose=True, caching=True)

#Reading and setting up config parser
config = ConfigParser.ConfigParser()
config.readfp(open('configs/macros.cfg'))

#setting up device,model name and loss types.
training_model = 'bilstm_dot'
_dataset = 'lcquad'
pointwise = False
_debug = False

#Loading relations file.
COMMON_DATA_DIR = 'data/data/common'
INTENTS = ['count', 'ask', 'list']
RDFTYPES = ['x', 'uri', 'none']

_dataset_specific_data_dir = 'data/data/%(dataset)s/' % {'dataset': _dataset}
_relations = aux.load_relation(COMMON_DATA_DIR)
_word_to_id = aux.load_word_list(COMMON_DATA_DIR)

# Model specific paramters
    # #Model specific paramters
if pointwise:
    training_config = 'pointwise'
else:
    training_config = 'pairwise'

parameter_dict = cl.runtime_parameters(dataset=_dataset,training_model=training_model,
                                         training_config=training_config,config_file='configs/macros.cfg')

if training_model == 'cnn_dot':
    parameter_dict['output_dim'] = int(config.get(training_model, 'output_dim'))

# Update parameters
parameter_dict['_dataset_specific_data_dir'] = _dataset_specific_data_dir
parameter_dict['_model_dir'] = './data/models/'

parameter_dict['corechainmodel'] = 'bilstm_dot'
parameter_dict['corechainmodelnumber'] = '20'

parameter_dict['intentmodel'] = 'bilstm_dense'
parameter_dict['intentmodelnumber'] = '5'

parameter_dict['rdftypemodel'] = 'bilstm_dense'
parameter_dict['rdftypemodelnumber'] = '6'

parameter_dict['rdfclassmodel'] = 'bilstm_dot'
parameter_dict['rdfclassmodelnumber'] = '9'


class QuestionAnswering:
    """
        Usage:

            qa = QuestionAnswering(parameter_dict, False, _word_to_id, device, True)
            q = np.random.randint(0, 1233, (542))
            p = np.random.randint(0, 123, (10, 55))
            print(qa._predict_corechain(q,p))
            print("intent: ", qa._predict_intent(q))
            print("rdftype: ", qa._predict_rdftype(q))
            print("rdfclass: ", qa._predict_rdfclass(q, p))
    """

    def __init__(self, parameters, pointwise, word_to_id, device, debug):

        self.parameters = parameters
        self.pointwise = pointwise
        self.debug = debug
        self.device = device
        self._word_to_id = word_to_id

        # Load models
        self._load_corechain_model()
        self._load_rdftype_model()
        self._load_rdfclass_model()
        self._load_intentmodel()

    def _load_corechain_model(self):

        # Initialize the model
        if self.parameters['corechainmodel'] == 'bilstm_dot':
            self.corechain_model = net.BiLstmDot(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        # Make the model path
        model_path = os.path.join(self.parameters['_model_dir'], 'core_chain')
        model_path = os.path.join(model_path, self.parameters['corechainmodel'])
        model_path = os.path.join(model_path, self.parameters['dataset'])
        model_path = os.path.join(model_path, self.parameters['corechainmodelnumber'])
        model_path = os.path.join(model_path, 'model.torch')

        self.corechain_model.load_from(model_path)

    def _load_rdfclass_model(self):

        # Initialize the model
        if self.parameters['rdfclassmodel'] == 'bilstm_dot':
            self.rdfclass_model = net.BiLstmDot(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        # Make the model path
        model_path = os.path.join(self.parameters['_model_dir'], 'rdf_class')
        model_path = os.path.join(model_path, self.parameters['rdfclassmodel'])
        model_path = os.path.join(model_path, self.parameters['dataset'])
        model_path = os.path.join(model_path, self.parameters['rdfclassmodelnumber'])
        model_path = os.path.join(model_path, 'model.torch')

        self.rdfclass_model.load_from(model_path)

    def _load_rdftype_model(self):
        # Initialize the model
        self.rdftype_model = net_rdftype.RdfTypeClassifier(_parameter_dict=self.parameters,
                                                           _word_to_id=self._word_to_id,
                                                           _device=self.device)

        # Make model path like:
        # ('model with accuracy ', 0.998, 'stored at', 'data/models/intent/bilstm_dense/lcquad/2/model.torch')
        model_path = os.path.join(self.parameters['_model_dir'], 'rdf_type')
        model_path = os.path.join(model_path, self.parameters['rdftypemodel'])
        model_path = os.path.join(model_path, self.parameters['dataset'])
        model_path = os.path.join(model_path, self.parameters['rdftypemodelnumber'])
        model_path = os.path.join(model_path, 'model.torch')

        self.rdftype_model.load_from(model_path)

    def _load_intentmodel(self):

        # Initialize the model
        self.intent_model = net_intent.IntentClassifier(_parameter_dict=self.parameters,
                                                        _word_to_id=self._word_to_id,
                                                        _device=self.device)

        # Make model path like:
        # ('model with accuracy ', 0.998, 'stored at', 'data/models/intent/bilstm_dense/lcquad/2/model.torch')
        model_path = os.path.join(self.parameters['_model_dir'], 'intent')
        model_path = os.path.join(model_path, self.parameters['intentmodel'])
        model_path = os.path.join(model_path, self.parameters['dataset'])
        model_path = os.path.join(model_path, self.parameters['intentmodelnumber'])
        model_path = os.path.join(model_path, 'model.torch')

        self.intent_model.load_from(model_path)

    def _predict_corechain(self, _q, _p):
        """
            Given a datapoint (question, paths) encoded in  embedding_vocab,
                run the model's predict and find the best corechain.

            _q: (<var len>)
            _p: (100/500, <var len>)

            returns score: (100/500)
        """

        # Pad questions
        Q = np.zeros((len(_p), self.parameters['max_length']))
        Q[:, :min(len(_q), self.parameters['max_length'])] = \
            np.repeat(_q[np.newaxis, :min(len(_q), self.parameters['max_length'])], repeats=len(_p), axis=0)

        # Pad paths
        P = np.zeros((len(_p), self.parameters['max_length']))
        for i in range(len(_p)):
            P[i, :min(len(_p[i]), self.parameters['max_length'])] = \
                _p[i][:min(len(_p[i]), self.parameters['max_length'])]

        # Convert np to torch stuff
        Q = torch.tensor(Q, dtype=torch.long, device=self.device)
        P = torch.tensor(P, dtype=torch.long, device=self.device)

        if self.debug:
            print("Q: ", Q.shape, " P: ", P.shape)

            # We then pass them through a predict function and get a score array.
        score = self.corechain_model.predict(ques=Q, paths=P, device=self.device)

        return score.detach().cpu().numpy()

    def _predict_rdfclass(self, _q, _p):
        """
            Given a datapoint (question, paths) encoded in  embedding_vocab,
                run the model's predict and find the best corechain.

            _q: (<var len>)
            _p: (100/500, <var len>)

            returns score: (100/500)
        """

        # Pad questions
        Q = np.zeros((len(_p), self.parameters['max_length']))
        Q[:, :min(len(_q), self.parameters['max_length'])] = \
            np.repeat(_q[np.newaxis, :min(len(_q), self.parameters['max_length'])], repeats=len(_p), axis=0)

        # Pad paths
        P = np.zeros((len(_p), self.parameters['max_length']))
        for i in range(len(_p)):
            P[i, :min(len(_p[i]), self.parameters['max_length'])] = \
                _p[i][:min(len(_p[i]), self.parameters['max_length'])]

        # Convert np to torch stuff
        Q = torch.tensor(Q, dtype=torch.long, device=self.device)
        P = torch.tensor(P, dtype=torch.long, device=self.device)

        # We then pass them through a predict function and get a score array.

        score = self.rdfclass_model.predict(ques=Q, paths=P, device=self.device)

        return score.detach().cpu().numpy()

    def _predict_intent(self, _q):
        """
            Given a question, it runs a distribution over possible intents (ask/count/list)

            _q: (<var len>)

            returns: np.arr shape (3)
        """

        # Pad the question
        Q = np.zeros(self.parameters['max_length'])
        Q[:min(_q.shape[0], self.parameters['max_length'])] = _q[:min(_q.shape[0], self.parameters['max_length'])]

        data = {'ques_batch': Q.reshape(1, Q.shape[0])}

        # Get prediction
        score = self.intent_model.predict(data, self.device)

        return score.detach().cpu().numpy()

    def _predict_rdftype(self, _q):
        """
            Given a question, it runs a distribution over possible places where we attach an rdftype constraint
                (x/uri/none)

            _q: (<var len>)

            returns: np.arr shape (3)
        """

        # Pad the question
        Q = np.zeros(self.parameters['max_length'])
        Q[:min(_q.shape[0], self.parameters['max_length'])] = _q[:min(_q.shape[0], self.parameters['max_length'])]

        data = {'ques_batch': Q.reshape(1, Q.shape[0])}

        # Get prediction
        score = self.rdftype_model.predict(data, self.device)

        return score.detach().cpu().numpy()

def construct_paths(data, relations, gloveid_to_embeddingid, qald=False):
    """
    :param data: a data node of id_big_data
    relations : a dictionary which maps relation id to meta inforamtion like surface form, embedding id
    of surface form etc.
    :return: unpadded , continous id spaced question, positive path, negative paths

    @TODO: remove from here, and use dataloader version

    """

    question = np.asarray(data['uri']['question-id'])
    # questions = pad_sequences([question], maxlen=max_length, padding='post')

    # inverse id version of positive path and creating a numpy version
    positive_path_id = data['parsed-data']['path_id']
    no_positive_path = False
    if positive_path_id == [-1]:
        positive_path = np.asarray([-1])
        no_positive_path = True
    else:
        positive_path = []
        for path in positive_path_id:
            positive_path += [embeddings_interface.SPECIAL_CHARACTERS.index(path[0])]
            positive_path += relations[int(path[1:])][3].tolist()
        positive_path = np.asarray(positive_path)
    # padded_positive_path = pad_sequences([positive_path], maxlen=max_length, padding='post')

    # negative paths from id to surface form id
    negative_paths_id = data['uri']['hop-2-properties'] + data['uri']['hop-1-properties']
    negative_paths = []
    for neg_path in negative_paths_id:
        negative_path = []
        for path in neg_path:
            try:
                negative_path += [embeddings_interface.SPECIAL_CHARACTERS.index(path)]
            except ValueError:
                negative_path += relations[int(path)][3].tolist()
        negative_paths.append(np.asarray(negative_path))
    negative_paths = np.asarray(negative_paths)
    # negative paths padding
    # padded_negative_paths = pad_sequences(negative_paths, maxlen=max_length, padding='post')

    # explicitly remove any positive path from negative path
    negative_paths = dl.remove_positive_path(positive_path, negative_paths)

    # remap all the id's to the continous id space.

    # passing all the elements through vocab
    question = np.asarray([gloveid_to_embeddingid[key] for key in question])
    if not no_positive_path:
        positive_path = np.asarray([gloveid_to_embeddingid[key] for key in positive_path])
    for i in range(0, len(negative_paths)):
        # temp = []
        for j in xrange(0, len(negative_paths[i])):
            try:
                negative_paths[i][j] = gloveid_to_embeddingid[negative_paths[i][j]]
            except:
                negative_paths[i][j] = gloveid_to_embeddingid[0]
                # negative_paths[i] = np.asarray(temp)
                # negative_paths[i] = np.asarray([vocab[key] for key in negative_paths[i] if key in vocab.keys()])
    if qald:
        return question, positive_path, negative_paths, no_positive_path
    return question, positive_path, negative_paths

def prune_candidate_space(question, paths, k=None):
    """
        Boilerplate to reduce the number of valid paths.
        Note: path[0] is the correct path.
            Should we remove it? Should we not?

        As of now it returns an index
    """

    return np.arange(len(paths))

def create_sparql(log, data, embeddings_interface, embeddingid_to_gloveid, relations):
    """
        Creates a query graph from logs and sends it to sparql_constructor
            for getting a valid SPARQL query (or results) back.


        Query graph is a dict containing:
            best_path,
            intent,
            rdf_constraint,
            rdf_constraint_type,
            rdf_best_path

    :param log: dict made using answer_question function
    :param embeddings_interface: the file
    :param embeddingid_to_gloveid: reverse vocab dict
    :param relations: the relations dict
    :return: sparql query as string
    """
    query_graph = {}
    query_graph['intent'] = log['pred_intent']
    query_graph['best_path'] = log['pred_path']
    query_graph['rdf_constraint_type'] = log['pred_rdf_type']
    query_graph['rdf_best_path'] = log['pred_rdf_class']
    query_graph['entities'] = data['parsed-data']['entity']
    query_graph['rdf_constraint'] = False if log['pred_rdf_type'] == 'none' else True

    return sparql_constructor.convert(_graph=query_graph, relations=relations,
                                        embeddings_interface=embeddings_interface,
                                        embeddingid_to_gloveid=embeddingid_to_gloveid)


def corechain_prediction(question, paths, positive_path, negative_paths, no_positive_path):
    '''
        Why is path needed ?
    '''

    # Remove if adding to class
    global qa

    mrr = 0
    best_path = ''
    path_predicted_correct = False

    if no_positive_path and len(negative_paths) == 0:
        '''
            There exists no positive path and also no negative paths
                Why does this quest exists ? 
                    > Probably in qald
        '''
        print("The code should not have been herr. There is no warning. RUN!!!!!!!!")
        raise ValueError

    elif not no_positive_path and len(negative_paths) == 0:
        '''
            There exists a positive path and there exists no negative path
        '''
        best_path = positive_path
        mrr = 1
        path_predicted_correct = True

    elif no_positive_path and len(negative_paths) != 0:
        '''
            There exists no correct/true path and there are few negative paths.
        '''
        output = qa._predict_corechain(question, paths)
        best_path_index = np.argmax(output)
        best_path = paths[best_path_index]

    elif not no_positive_path and len(negative_paths) != 0:
        '''
            There exists positive path and also negative paths
            path = positive_path + negative_paths    
        '''
        output = qa._predict_corechain(question, paths)
        best_path_index = np.argmax(output)
        best_path = paths[best_path_index]

        # Calculate mrr here
        mrr = 0
        if best_path_index == 0:
            path_predicted_correct = True

        mrr_output = np.argsort(output)[::-1]
        mrr_output = mrr_output.tolist()
        mrr = mrr_output.index(0) + 1.0

        if mrr != 0:
            mrr = 1.0 / mrr

    else:
        print("The code should not have been herr. There is no warning. RUN!!!!!!!!")
        raise ValueError

    return mrr, best_path, path_predicted_correct


def answer_question(qa, index, data, gloveid_to_embeddingid, embeddingid_to_gloveid, relations, parameter_dict):
    """
        Uses everything to do everyhing for one data instance (one question, subgraph etc).
    """

    log = {}
    log['question'] = None
    log['true_path'] = None
    log['true_intent'] = None
    log['true_rdf_type'] = None
    log['true_rdf_class'] = None
    log['pred_path'] = None
    log['pred_intent'] = None
    log['pred_rdf_type'] = None
    log['pred_rdf_class'] = None

    metrics = {}

    question, positive_path, negative_paths, no_positive_path = dl.construct_paths(data, qald=True,
                                                                                   relations=relations,
                                                                                   gloveid_to_embeddingid=gloveid_to_embeddingid)
    log['question'] = question

    '''
        @some hack
        if the dataset is LC-QUAD and data['pop'] 
            is false then the positive path has been forcefully inserted and needs to be removed.
    '''
    if parameter_dict['dataset'] == 'lcquad':
        try:
            if data['pop'] == False:
                no_positive_path = True
        except KeyError:
            pass

    # ##############################################
    """
        Core chain prediction
    """
    # ##############################################
    if no_positive_path:
        '''
            There is no positive path, maybe we do something intelligent
        '''
        log['true_path'] = [-1]
        nps = [n.tolist() for n in negative_paths]
        paths = nps
        index_selected_paths = prune_candidate_space(question, paths, parameter_dict['prune_corechain_candidates'])

    else:

        pp = [positive_path.tolist()]
        nps = [n.tolist() for n in negative_paths]
        paths = pp + nps
        if parameter_dict['prune_corechain_candidates']:
            index_selected_paths = prune_candidate_space(question, paths, parameter_dict['prune_corechain_candidates'])

            if index_selected_paths[-1] == 0:
                #  Counts the number of times just using  word2vec similarity, the best path came the most similar.
                # This will only work if CANDIDATE_SPACE is not none.
                metrics['word_vector_accuracy_counter'] = 1
        else:
            index_selected_paths = prune_candidate_space(question, paths, len(paths))

        log['true_path'] = pp[0]

    # Put the pruning index over the paths
    paths = [paths[i] for i in index_selected_paths]
    '''
        Converting paths to numpy array
    '''
    for i in range(len(paths)):
        paths[i] = np.asarray(paths[i])
    paths = np.asarray(paths)

    cc_mrr, best_path, cc_acc = corechain_prediction(question,
                                                     paths, positive_path,
                                                     negative_paths, no_positive_path)

    log['pred_path'] = best_path
    metrics['core_chain_accuracy_counter'] = cc_acc
    metrics['core_chain_mrr_counter'] = cc_mrr
    metrics['num_paths'] = len(paths)

    # ##############################################
    """
        Intent, rdftype prediction

        Straightforward.

        Metrics: accuracy
    """
    # ##############################################
    # Get intent
    intent_pred = np.argmax(qa._predict_intent(question))
    intent_true = np.argmax(net_intent.get_y(data))
    intent_acc = 1 if intent_pred == intent_true else 0
    metrics['intent_accuracy_counter'] = intent_acc
    intent = INTENTS[intent_pred]

    log['true_intent'] = INTENTS[intent_true]
    log['pred_intent'] = INTENTS[intent_pred]

    # Get rdftype
    rdftype_pred = np.argmax(qa._predict_rdftype(question))
    rdftype_true = np.argmax(net_rdftype.get_y(data))
    rdftype_acc = 1 if rdftype_pred == rdftype_true else 0
    metrics['rdftype_accuracy_counter'] = rdftype_acc
    rdftype = RDFTYPES[rdftype_pred]

    log['true_rdf_type'] = RDFTYPES[rdftype_true]
    log['pred_rdf_type'] = RDFTYPES[rdftype_pred]

    # ##############################################
    """
        RDF class prediction.

            do this only if we need to, based on the prediction of rdftype model.
    """
    # ##############################################

    # Add dummy rdfclass logs and metrics
    log['true_rdf_class'] = None
    log['pred_rdf_class'] = None
    metrics['rdfclass_accuracy_counter'] = None

    if rdftype == "none":

        pass

    else:
        """
            We do need an rdf constraint.
            We let the rdf class model (ranker) choose between both x and uri paths, 
                and the rdf type model is just used to see if we need paths at all.
        """
        rdf_candidates = sparql_constructor.rdf_type_candidates(data, best_path,
                                                                gloveid_to_embeddingid,
                                                                relations,
                                                                embeddingid_to_gloveid)

        if rdf_candidates:

            rdf_candidate_pred = qa._predict_rdfclass(_q=question, _p=rdf_candidates)
            best_rdf_path = rdf_candidates[np.argmax(rdf_candidate_pred)]

        else:

            # No candidates found
            best_rdf_path = []

        # @TODO: as of now we don't have ground truth so we add a 0 in metrics and 0 in log
        log['true_rdf_class'] = 0
        log['pred_rdf_class'] = best_rdf_path
        metrics['rdfclass_accuracy_counter'] = 0

    return log, metrics

def sparql_answer(sparql):
    test_answer = []
    interface_test_answer = dbp.get_answer(sparql)
    for key in interface_test_answer:
        test_answer = test_answer + interface_test_answer[key]
    return list(set(test_answer))

def _evaluate_sparqls_(test_sparql, true_sparql, type, ground_type):
    # @TODO: If the type of test and true are differnt code would return an error.
    """
        Fmeasure for ask and count are 0/1.
        Also assumes the variable to be always uri.
        :param test_sparql: SPARQL generated by the pipeline
        :param true_sparql: True SPARQL
        :param type: COUNT/ASK/LIST
        :return: f1,precision,recall
    """
    '''
        First evaluate based on type. If the type prediction is wrong. Don't proceded. The f,p,r =0
    '''
    if type != ground_type:
        return 0.0 ,0.0 ,0.0

    if type == "list":
        test_answer = sparql_answer(test_sparql)
        true_answer = sparql_answer(true_sparql)
        total_retrived_resutls = len(test_answer)
        total_relevant_resutls = len(true_answer)
        common_results = total_retrived_resutls - len(list(set(test_answer ) -set(true_answer)))
        if total_retrived_resutls == 0:
            precision = 0
        else:
            precision = common_results *1.0 /total_retrived_resutls
        if total_relevant_resutls == 0:
            recall = 0
        else:
            recall = common_results *1.0 /total_relevant_resutls
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = (2.0 * (precision * recall)) / (precision + recall)
        return f1 ,precision ,recall

    if type == "count":
        count_test = sparql_answer(test_sparql)
        count_true = sparql_answer(true_sparql)
        if count_test == count_true:
            return 1.0 ,1.0 ,1.0
        else:
            return 0.0 ,0.0 ,1.0

    if type == "ask":
        if dbp.get_answer(test_sparql) == dbp.get_answer(true_sparql):
            return 1.0 ,1.0 ,1.0
        else:
            return 0.0 ,0.0 ,1.0


def evaluate(_logging):
    """
        After an entire run, pass the logging here.

        It plots the accuracy, and intelligently computes all the metrics.

        @TODO: handle rdf class metrics, and supervision with answers
    """

    corechain_acc, corechain_mrr = [], []
    intent_acc, rdf_type_acc, rdf_class_acc = [], [], []
    overall_p, overall_r, overall_f = [], [], []

    pbar = ProgressBar()

    for log in pbar(_logging['runtime']):
        corechain_acc.append(log['metrics']['core_chain_accuracy_counter'])
        corechain_mrr.append(log['metrics']['core_chain_mrr_counter'])
        intent_acc.append(log['metrics']['intent_accuracy_counter'])
        rdf_type_acc.append(log['metrics']['rdftype_accuracy_counter'])

        if log['log']['true_rdf_class']:
            rdf_class_acc.append(log['metrics']['rdfclass_accuracy_counter'])

        # Get f p r
        f, p, r = _evaluate_sparqls_(test_sparql=log['pred_sparql'],
                                     true_sparql=log['true_sparql'],
                                     type=log['log']['pred_intent'],
                                     ground_type=log['log']['true_intent'])
        overall_p.append(p)
        overall_r.append(r)
        overall_f.append(f)

    print("Corechain accuracy:\t", np.mean(corechain_acc))
    print("Corechain mean rr:\t", np.mean(corechain_mrr))
    print("Intent Accuracy:\t", np.mean(intent_acc))
    print("RDF type Accuracy:\t", np.mean(rdf_type_acc))
    print("RDF class accuracy:\t", np.mean(rdf_class_acc), "  over len ", len(rdf_class_acc))
    print("Overall Precision:\t", np.mean(overall_p))
    print("Overall Recall:\t", np.mean(overall_r))
    print("Overall F1Score:\t", np.mean(overall_f))


if __name__ == "__main__":
    TEMP = aux.data_loading_parameters(_dataset, parameter_dict, runtime=True)

    _dataset_specific_data_dir, _model_specific_data_dir, _file, \
    _max_sequence_length, _neg_paths_per_epoch_train, \
    _neg_paths_per_epoch_validation, _training_split, _validation_split, _index = TEMP

    _data, _gloveid_to_embeddingid, _vectors = dl.create_dataset_runtime(file=_file, _dataset=_dataset,
                                                                         _dataset_specific_data_dir=_dataset_specific_data_dir,
                                                                         split_point=.80)

    parameter_dict['vectors'] = _vectors

    # For interpretability's sake
    gloveid_to_embeddingid, embeddingid_to_gloveid, word_to_gloveid, \
    gloveid_to_word = aux.load_embeddingid_gloveid()

    """
        Different counters and metrics to store accuracy of diff modules

            Core chain accuracy counter counts the number of time the core chain predicated is same as 
            positive path. This also includes for ask query.
            The counter might confuse the property and the ontology. 

            Similar functionality with rdf_type and intent

            **word vector accuracy counter**: 
                Counts the number of times just using  word2vec similarity, 
                the best path came the most similar. 
                This will only work if CANDIDATE_SPACE is not none.

    """

    '''
        c_flag  is true if the core_chain was correctly predicted. 
        same is the case for i_flag and r_flag, rt_flag (correct candidate for rdf type)
    '''
    c_flag, i_flag, r_flag, rt_flag = False, False, False, False

    '''
        Stores tuple of (fmeasure,precision,recall)
    '''
    results = []

    Logging = parameter_dict.copy()
    Logging['runtime'] = []

    qa = QuestionAnswering(parameter_dict, pointwise, _word_to_id, device, False)

    # Some logs which run during runtime, not after.
    core_chain_acc_log = []
    core_chain_mrr_log = []

    startindex = 0
    for index, data in enumerate(_data[startindex:]):
        index += startindex

        log, metrics = answer_question(qa=qa,
                                       index=index,
                                       data=data,
                                       gloveid_to_embeddingid=_gloveid_to_embeddingid,
                                       embeddingid_to_gloveid=embeddingid_to_gloveid,
                                       relations=_relations,
                                       parameter_dict=parameter_dict)

        #     log, metrics = answer_question(qa=None,
        #                                    index=None,
        #                                    data=None,
        #                                    gloveid_to_embeddingid=None,
        #                                    embeddingid_to_gloveid=None,
        #                                    relations=None,
        #                                    parameter_dict=None)

        sparql = create_sparql(log=log,
                               data=data,
                               embeddings_interface=embeddings_interface,
                               embeddingid_to_gloveid=embeddingid_to_gloveid,
                               relations=_relations)

        # metrics = eval(data, log, metrics)

        # Update logs
        Logging['runtime'].append({'log': log, 'metrics': metrics,
                                   'pred_sparql': sparql, 'true_sparql': data['parsed-data']['sparql_query']})

        # Update metrics
        core_chain_acc_log.append(metrics['core_chain_accuracy_counter'])
        core_chain_mrr_log.append(metrics['core_chain_mrr_counter'])

        # Make shit interpretable
        question = aux.id_to_word(log['question'], gloveid_to_word, embeddingid_to_gloveid, remove_pad=True)
        true_path = aux.id_to_word(log['true_path'], gloveid_to_word, embeddingid_to_gloveid, remove_pad=True)
        pred_path = aux.id_to_word(log['pred_path'], gloveid_to_word, embeddingid_to_gloveid, remove_pad=True)

        print("#%s" % index, "\t\bAcc: ", np.mean(core_chain_acc_log))

        #     print("\t\bQues: ", question)
        #     print("\t\bTPath: ", true_path, "\n\t\bPPath: ", pred_path)

        #     print("\t\bTIntent: ", log['true_intent'])
        #     print("\t\bPIntent: ", log['pred_intent'])
        #     print("\t\bPRdftype: ", log['true_rdf_type'])
        #     print("\t\bTRdftype: ", log['pred_rdf_type'])
        #     print("\t\bPRdfclass: ", log['true_rdf_class'])
        #     print("\t\bTRdfclass: ", log['pred_rdf_class'])

        #     print("")
        #     pprint(log)
        #     print("")
        pprint(metrics)
        #     print("\n",sparql)
        print("\n################################\n")

    evaluate(Logging)
