from __future__ import print_function

from utils import query_graph_to_sparql as sparql_constructor
from utils import dbpedia_interface as db_interface
from utils import embeddings_interface
from configs import config_loader as cl
from utils import natural_language_utilities as nlutils
import network_rdftype as net_rdftype
import network_intent as net_intent
import data_loader as dl
import auxiliary as aux
import network as net

import os
import sys
import torch
import pickle
import traceback
import numpy as np
from pprint import pprint
from progressbar import ProgressBar

if sys.version_info[0] == 3: import configparser as ConfigParser
else: import ConfigParser

# Loading relations file.
COMMON_DATA_DIR = 'data/data/common'
INTENTS = ['count', 'ask', 'list']
RDFTYPES = ['x', 'uri', 'none']
# params for ULMFit
# parameter_dict['intentmodel'] = 'bilstm_dense'
# parameter_dict['intentmodelnumber'] = '16'
#
# parameter_dict['rdftypemodel'] = 'bilstm_dense'
# parameter_dict['rdftypemodelnumber'] = '12'
#
# parameter_dict['rdfclassmodel'] = 'bilstm_dot'
# parameter_dict['rdfclassmodelnumber'] = '16'

glove_id_sf_to_glove_id_rel = dl.create_relation_lookup_table('data/data/common')

class QuestionAnswering:
    """
        Usage:

            qa = QuestionAnswering(parameter_dict, False, _word_to_id, device, True)
            q = np.rancorechainmodeldom.randint(0, 1233, (542))
            p = np.random.randint(0, 123, (10, 55))
            print(qa._predict_corechain(q,p))
            print("intent: ", qa._predict_intent(q))
            print("rdftype: ", qa._predict_rdftype(q))
            print("rdfclass: ", qa._predict_rdfclass(q, p))
    """

    def __init__(self, parameters, pointwise, word_to_id, device, _dataset,debug):

        self.parameters = parameters
        self.pointwise = pointwise
        self.debug = debug
        self.device = device
        self._word_to_id = word_to_id

        # Load models
        # self.parameters['dataset'] = 'transfer-b'
        self._load_corechain_model()
        '''
            since all auxilary components perform really bad if just trained on QALD
            We always use ones trained on LC-QuAD.'        
        '''
        self.parameters['dataset'] = 'lcquad'
        self._load_rdftype_model()
        self._load_rdfclass_model()
        self._load_intentmodel()
        self.parameters['dataset'] = _dataset

    def _load_corechain_model(self):

        # Initialize the model
        m = self.parameters['corechainmodel']
        # self.parameters['corechainmodel'] = 'slotptrortho'
        # self.parameters['bidirectional'] = False
        if self.parameters['corechainmodel'] == 'bilstm_dot':
            self.corechain_model = net.BiLstmDot(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        if self.parameters['corechainmodel'] == 'bilstm_densedot':
            self.corechain_model = net.BiLstmDenseDot(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        if self.parameters['corechainmodel'] == 'slotptr':
            self.corechain_model = net.QelosSlotPointerModel(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        if self.parameters['corechainmodel'] == 'slotptr_common_encoder':
            self.corechain_model = net.QelosSlotPointerModel_common_encoder(_parameter_dict=self.parameters,
                                                             _word_to_id=self._word_to_id,
                                                             _device=self.device, _pointwise=self.pointwise,
                                                             _debug=self.debug)
        if self.parameters['corechainmodel'] == 'slotptrortho':
            self.corechain_model = net.QelosSlotPointerModelOrthogonal(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        if self.parameters['corechainmodel'] == 'reldet':
            self.corechain_model = net.RelDetection(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        if self.parameters['corechainmodel'] == 'decomposable_attention':
            self.corechain_model = net.DecomposableAttention(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)
        if self.parameters['corechainmodel'] == 'cnn_dot':
            self.corechain_model = net.CNNDot(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        if self.parameters['corechainmodel'] == 'bilstm_dot_multiencoder':
            self.corechain_model = net.BiLstmDot_multiencoder(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        if self.parameters['corechainmodel'] == 'ulmfit_slotptr':
            self.corechain_model = net.ULMFITQelosSlotPointerModel(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)

        if self.parameters['corechainmodel'] == 'bert':
            ### This needs to change
            self.corechain_model = net.Bert_Scorer(_parameter_dict = parameter_dict,
                    _word_to_id=_word_to_id,
                    _device=device,
                    _pointwise=pointwise,
                    _debug=False)

        if self.parameters['corechainmodel'] == 'slotptr_randomvec':
            self.corechain_model = net.QelosSlotPointerModelRandomVec(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                 _device=self.device, _pointwise=self.pointwise, _debug=self.debug)


        if self.parameters['corechainmodel'] == 'bert_slotptr':
            self.corechain_model = net.Bert_Scorer_slotptr(_parameter_dict=self.parameters,
                                          _word_to_id=self._word_to_id,
                                          _device=self.device,
                                          _pointwise=self.pointwise,
                                          _debug=self.debug)


        # Make the model path
        model_path = os.path.join(self.parameters['_model_dir'], 'core_chain')
        if self.pointwise:
            model_path = os.path.join(model_path, self.parameters['corechainmodel']+'_pointwise')
        else:
            model_path = os.path.join(model_path, self.parameters['corechainmodel'])
            # model_path = os.path.join(model_path, self.parameters['slotptr_common_encoder'])
        model_path = os.path.join(model_path, self.parameters['dataset'])
        # model_path = os.path.join(model_path, "transfer-b")
        model_path = os.path.join(model_path, self.parameters['corechainmodelnumber'])
        model_path = os.path.join(model_path, 'model.torch')

        self.corechain_model.load_from(model_path)

        self.parameters['corechainmodel'] = m
        self.parameters['bidirectional'] = True

    def _load_rdfclass_model(self):

        # Initialize the model
        if self.parameters['rdfclassmodel'] == 'bilstm_dot':
            self.rdfclass_model = net.BiLstmDot(_parameter_dict=self.parameters, _word_to_id=self._word_to_id,
                                                _device=self.device, _pointwise=False, _debug=self.debug)

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

    def _predict_corechain_old(self, _q, _p, _p1 = None , _p2 = None, _p1_randomvec = None, _p2_randomvec = None):
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
        if _p1:
            P1 = np.zeros((len(_p), self.parameters['max_length']))
            P2 = np.zeros((len(_p), self.parameters['max_length']))
        for i in range(len(_p)):
            P[i, :min(len(_p[i]), self.parameters['max_length'])] = _p[i][:min(len(_p[i]), self.parameters['max_length'])]

        if _p1_randomvec:
            P1_randomvec = np.zeros((len(_p), self.parameters['max_length']))
            P2_randomvec = np.zeros((len(_p), self.parameters['max_length']))

        if _p1:
            # print(_p1)
            for i in range(len(_p)):
                # print(type(_p1[i]),_p1[i],_p1[:5])
                P1[i, :min(len(_p1[i]), self.parameters['max_length'])] = _p1[i][
                                                                        :min(len(_p1[i]), self.parameters['max_length'])]
                P2[i, :min(len(_p2[i]), self.parameters['max_length'])] = _p2[i][
                                                                          :min(len(_p2[i]),
                                                                                   self.parameters['max_length'])]
            P1 = torch.tensor(P1, dtype=torch.long, device=self.device)
            P2 = torch.tensor(P2, dtype=torch.long, device=self.device)

            if self.parameters['corechainmodel'] == 'slotptr' or self.parameters['corechainmodel'] == 'slotptr_randomvec'\
                    or self.parameters['corechainmodel'] == 'bert_slotptr':
                P1 = P1[:,:self.parameters['relsp_pad']]
                P2 = P2[:,:self.parameters['relsp_pad']]
            else:
                P1 = P1[:, :self.parameters['relrd_pad']]
                P2 = P2[:, :self.parameters['relrd_pad']]

        if _p1_randomvec:
            # print(_p1)
            for i in range(len(_p)):
                # print(type(_p1[i]),_p1[i],_p1[:5])
                P1_randomvec[i, :min(len(_p1[i]), self.parameters['max_length'])] = _p1_randomvec[i][
                                                                        :min(len(_p1_randomvec[i]), self.parameters['max_length'])]
                P2_randomvec[i, :min(len(_p2[i]), self.parameters['max_length'])] = _p2_randomvec[i][
                                                                          :min(len(_p2_randomvec[i]),
                                                                                   self.parameters['max_length'])]
            P1_randomvec = torch.tensor(P1_randomvec, dtype=torch.long, device=self.device)
            P2_randomvec = torch.tensor(P2, dtype=torch.long, device=self.device)

            P1_randomvec = P1_randomvec[:, :self.parameters['relrd_pad']]
            P2_randomvec = P2_randomvec[:, :self.parameters['relrd_pad']]


        # Convert np to torch stuff
        Q = torch.tensor(Q, dtype=torch.long, device=self.device)
        P = torch.tensor(P, dtype=torch.long, device=self.device)
        P = P[:, :self.parameters['rel_pad']]
        # if self.debug:
            # print("Q: ", Q.shape, " P: ", P.shape)

            # We then pass them through a predict function and get a score array.
        if self.parameters['corechainmodel'] == 'slotptr' or self.parameters['corechainmodel'] == 'reldet' or \
                self.parameters['corechainmodel'] == 'bert_slotptr':
            # print("path rel 1 main ", P1)
            # print("path rel 2 main ", P2)
            # print("path rel 2 main ", P1.shape)
            # print("path rel 2 main ", P2.shape)

            score = self.corechain_model.predict(ques=Q, paths=P, paths_rel1=P1, paths_rel2=P2, device=self.device)
            #Visual stuff.
            # score,attention_score = self.corechain_model.predict(ques=Q, paths=P, paths_rel1=P1, paths_rel2=P2, device=self.device,attention_value=True)
            # score1 = attention_score.squeeze(-1)[0, :, 0]
            # score2 = attention_score.squeeze(-1)[0, :, 1]
            # return score.detach().cpu().numpy(), score1.detach().cpu().numpy(), score2.detach().cpu().numpy()
        elif self.parameters['corechainmodel'] == 'slotptr_randomvec':
            score = self.corechain_model.predict(ques=Q, paths=P, paths_rel1=P1, paths_rel1_randomvec=P1_randomvec,
                    paths_rel2=P2, paths_rel2_randomvec=P2_randomvec, device=self.device)
        else:
            score = self.corechain_model.predict(ques=Q, paths=P, device=self.device)
        return score.detach().cpu().numpy()


    def _predict_corechain(self, _q, _p, _p1 = None , _p2 = None, _p1_randomvec = None, _p2_randomvec = None):
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
        if _p1:
            P1 = np.zeros((len(_p), self.parameters['max_length']))
            P2 = np.zeros((len(_p), self.parameters['max_length']))
        for i in range(len(_p)):
            P[i, :min(len(_p[i]), self.parameters['max_length'])] = _p[i][:min(len(_p[i]), self.parameters['max_length'])]

        if _p1_randomvec:
            P1_randomvec = np.zeros((len(_p), self.parameters['max_length']))
            P2_randomvec = np.zeros((len(_p), self.parameters['max_length']))

        if _p1:
            # print(_p1)
            for i in range(len(_p)):
                # print(type(_p1[i]),_p1[i],_p1[:5])
                P1[i, :min(len(_p1[i]), self.parameters['max_length'])] = _p1[i][
                                                                        :min(len(_p1[i]), self.parameters['max_length'])]
                P2[i, :min(len(_p2[i]), self.parameters['max_length'])] = _p2[i][
                                                                          :min(len(_p2[i]),
                                                                                   self.parameters['max_length'])]



            if self.parameters['corechainmodel'] == 'slotptr' or \
                    self.parameters['corechainmodel'] == 'slotptr_randomvec' or self.parameters['corechainmodel'] == 'bert_slotptr':
                P1 = P1[:,:self.parameters['relsp_pad']]
                P2 = P2[:,:self.parameters['relsp_pad']]
            else:
                P1 = P1[:, :self.parameters['relrd_pad']]
                P2 = P2[:, :self.parameters['relrd_pad']]


            # P1 = torch.tensor(P1, dtype=torch.long, device=self.device)
            # P2 = torch.tensor(P2, dtype=torch.long, device=self.device)



        if _p1_randomvec:
            # print(_p1)
            for i in range(len(_p)):
                # print(type(_p1[i]),_p1[i],_p1[:5])
                P1_randomvec[i, :min(len(_p1[i]), self.parameters['max_length'])] = _p1_randomvec[i][
                                                                        :min(len(_p1_randomvec[i]), self.parameters['max_length'])]
                P2_randomvec[i, :min(len(_p2[i]), self.parameters['max_length'])] = _p2_randomvec[i][
                                                                          :min(len(_p2_randomvec[i]),
                                                                                   self.parameters['max_length'])]

            P1_randomvec = P1_randomvec[:, :self.parameters['relrd_pad']]
            P2_randomvec = P2_randomvec[:, :self.parameters['relrd_pad']]

            # P1_randomvec = torch.tensor(P1_randomvec, dtype=torch.long, device=self.device)
            # P2_randomvec = torch.tensor(P2_randomvec, dtype=torch.long, device=self.device)




        # Tensorize things here

        # Convert np to torch stuff
        P = P[:, :self.parameters['rel_pad']]


        #Check what variables are None and which are not none.
        if not _p1_randomvec:
            P1_randomvec,P2_randomvec = None, None
        if not _p1:
            P1,P2 = None,None

        def distribute_it(np_array, k):
            # print(len(np_array))
            return np.array_split(np_array[:-1], k, axis=0)


        distribute = True
        k = 10

        if len(Q) < k+1:
            distribute = False
        if distribute:

            print("in distributed setting")
            if _p1_randomvec:
                Q_dist, P_dist, P1_dist, P2_dist, P1_randomvec_dist, P2_randomvec_dist = distribute_it(Q,k), \
                                                                                         distribute_it(P,k), \
                                                                                         distribute_it(P1,k), \
                                                                                         distribute_it(P2,k), \
                                                                                         distribute_it(P1_randomvec,k),\
                                                                                         distribute_it(P2_randomvec,k)

                temp_score = []
                for q,p,p1,p2,p1_rv,p2_rv in zip(Q_dist,P_dist,P1_dist,P2_dist,P1_randomvec_dist,P2_randomvec_dist):
                    temp_score.append(self.tensorized_Score(q,p,p1,p1_rv,p2,p2_rv))
            if not _p1_randomvec and _p1:
                Q_dist, P_dist, P1_dist, P2_dist = distribute_it(Q, k), \
                                                                                         distribute_it(P, k), \
                                                                                         distribute_it(P1, k), \
                                                                                         distribute_it(P2, k)

                temp_score = []
                for q, p, p1, p2 in zip(Q_dist, P_dist, P1_dist, P2_dist):
                    temp_score.append(self.tensorized_Score(q, p, p1, None, p2, None))

            if not _p1_randomvec and not _p1:
                Q_dist, P_dist = distribute_it(Q, k), \
                                                   distribute_it(P, k)

                temp_score = []
                for q, p in zip(Q_dist, P_dist):
                    temp_score.append(self.tensorized_Score(q, p, None, None, None, None))

            final_score = []
            for scores in temp_score:
                for s in scores:
                    final_score.append(s)

            return np.asarray(final_score)



        else:
            return self.tensorized_Score(Q,P,P1=P1,P1_randomvec=P1_randomvec,P2=P2,P2_randomvec=P2_randomvec)
        # # Q = torch.tensor(Q, dtype=torch.long, device=self.device)
        # # P = torch.tensor(P, dtype=torch.long, device=self.device)
        # # if self.debug:
        #     # print("Q: ", Q.shape, " P: ", P.shape)
        #
        #     # We then pass them through a predict function and get a score array.
        # if self.parameters['corechainmodel'] == 'slotptr' or self.parameters['corechainmodel'] == 'reldet':
        #     # print("path rel 1 main ", P1)
        #     # print("path rel 2 main ", P2)
        #     # print("path rel 2 main ", P1.shape)
        #     # print("path rel 2 main ", P2.shape)
        #
        #     score = self.corechain_model.predict(ques=Q, paths=P, paths_rel1=P1, paths_rel2=P2, device=self.device)
        #     #Visual stuff.
        #     # score,attention_score = self.corechain_model.predict(ques=Q, paths=P, paths_rel1=P1, paths_rel2=P2, device=self.device,attention_value=True)
        #     # score1 = attention_score.squeeze(-1)[0, :, 0]
        #     # score2 = attention_score.squeeze(-1)[0, :, 1]
        #     # return score.detach().cpu().numpy(), score1.detach().cpu().numpy(), score2.detach().cpu().numpy()
        # elif self.parameters['corechainmodel'] == 'slotptr_randomvec':
        #     score = self.corechain_model.predict(ques=Q, paths=P, paths_rel1=P1, paths_rel1_randomvec=P1_randomvec,
        #             paths_rel2=P2, paths_rel2_randomvec=P2_randomvec, device=self.device)
        # else:
        #     score = self.corechain_model.predict(ques=Q, paths=P, device=self.device)
        # return score.detach().cpu().numpy()


    def tensorized_Score(self,Q,P,P1=None,P1_randomvec=None,P2=None,P2_randomvec=None):

        # with torch.no_grad:
            # Tensorize vectors:





        Q = torch.tensor(Q, dtype=torch.long, device=self.device)
        P = torch.tensor(P, dtype=torch.long, device=self.device)
        # P = P[:, :self.parameters['rel_pad']]

        if type(P1) != type(None):
            # Then P2 also exists
            P1 = torch.tensor(P1, dtype=torch.long, device=self.device)
            P2 = torch.tensor(P2, dtype=torch.long, device=self.device)

        if type(P1_randomvec) != type(None):
            # Then P2 randomvec also exists
            P1_randomvec = torch.tensor(P1_randomvec, dtype=torch.long, device=self.device)
            P2_randomvec = torch.tensor(P2_randomvec, dtype=torch.long, device=self.device)

        # Send it to the module and expect some scores
        if self.parameters['corechainmodel'] == 'slotptr' or self.parameters['corechainmodel'] == 'reldet' or self.parameters['corechainmodel'] == 'bert_slotptr':
            score = self.corechain_model.predict(ques=Q, paths=P, paths_rel1=P1, paths_rel2=P2, device=self.device)

        elif self.parameters['corechainmodel'] == 'slotptr_randomvec':
            score = self.corechain_model.predict(ques=Q, paths=P, paths_rel1=P1, paths_rel1_randomvec=P1_randomvec,
                                                 paths_rel2=P2, paths_rel2_randomvec=P2_randomvec,
                                                 device=self.device)
        else:
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
    positive_path_id = data['parsed-data']['path']
    no_positive_path = False
    print("**", positive_path_id)
    if positive_path_id == -1:
        positive_path = np.asarray([-1])
        no_positive_path = True
    else:
        positive_path = []
        for p in positive_path_id:
            if p in ['+', '-']:
                positive_path += vocabularize_relation(p)
            else:
                positive_path += relations[int(p)][3].tolist()

        positive_path = np.asarray(positive_path)
    # padded_positive_path = pad_sequences([positive_path], maxlen=max_length, padding='post')

    # negative paths from id to surface form id
    negative_paths_id = data['uri']['hop-2-properties'] + data['uri']['hop-1-properties']
    negative_paths = []
    for neg_path in negative_paths_id:
        negative_path = []
        for path in neg_path:
            if path in embeddings_interface.SPECIAL_CHARACTERS:
                negative_path += vocabularize_relation(path)
            else:
                negative_path += relations[int(path)][3].tolist()
        negative_paths.append(np.asarray(negative_path))
    negative_paths = np.asarray(negative_paths)
    # negative paths padding
    # padded_negative_paths = pad_sequences(negative_paths, maxlen=max_length, padding='post')

    # explicitly remove any positive path from negative path
    negative_paths = dl.remove_positive_path(positive_path, negative_paths)

    # remap all the id's to the continous id space.

    # passing all the elements through vocab
    '''
        Legacy stuff.
        This was a mapping between glove and embedding id. For now we are nit using it. 
    '''
    # question = np.asarray([gloveid_to_embeddingid[key] for key in question])
    # if not no_positive_path:
    #     positive_path = np.asarray([gloveid_to_embeddingid[key] for key in positive_path])
    # for i in range(0, len(negative_paths)):
    #     # temp = []
    #     for j in range(0, len(negative_paths[i])):
    #         try:
    #             negative_paths[i][j] = gloveid_to_embeddingid[negative_paths[i][j]]
    #         except:
    #             negative_paths[i][j] = gloveid_to_embeddingid[0]
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


def create_sparql(log, data, embeddings_interface, relations):
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

    # return sparql_constructor.convert_runtime(_graph=query_graph)
    return sparql_constructor.convert(_graph=query_graph, relations=relations,
                                        embeddings_interface=embeddings_interface)


def create_rd_sp_paths(paths,no_reldet=False):
    special_char = [embeddings_interface.vocabularize(['+']), embeddings_interface.vocabularize(['-'])]
    dummy_path = [0]
    paths_rel1_sp = []
    paths_rel2_sp = []
    paths_rel1_rd = []
    paths_rel2_rd = []
    for p in paths:
        p1, p2 = dl.break_path(p, special_char)
        paths_rel1_sp.append(p1)
        '''
            >>>>IMPLEMENT THIS<<<<
            >>>>IMPLEMENT THIS<<<<
            >>>>IMPLEMENT THIS<<<<
            >>>>IMPLEMENT THIS<<<<
        '''
        if no_reldet:
            paths_rel1_rd.append(p1)
        else:
            paths_rel1_rd.append([dl.relation_table_lookup_reverse(p1,glove_id_sf_to_glove_id_rel)])
        if p2 is not None:
            paths_rel2_sp.append(p2)
            if no_reldet:
                paths_rel2_rd.append(p2)
            else:
                paths_rel2_rd.append([dl.relation_table_lookup_reverse(p2,glove_id_sf_to_glove_id_rel)])
        else:
            paths_rel2_sp.append(dummy_path)
            paths_rel2_rd.append(dummy_path)
    paths_rel1_sp = [np.asarray(o) for o in paths_rel1_sp]
    paths_rel2_sp = [np.asarray(o) for o in paths_rel2_sp]
    paths_rel1_rd = [np.asarray(o) for o in paths_rel1_rd]
    paths_rel2_rd = [np.asarray(o) for o in paths_rel2_rd]
    return paths_rel1_sp,paths_rel2_sp,paths_rel1_rd,paths_rel2_rd


def corechain_prediction(question, paths, positive_path, negative_paths, no_positive_path,model,quesans, verbal_question=""):
    '''
        Why is path needed ?
    '''

    # Remove if adding to class
    # global quesans

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
        # raise ValueError

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
        if model == 'reldet':
            _, _, paths_rel1_rd, paths_rel2_rd = create_rd_sp_paths(paths)
            # print("paths rel1 rd are loop1 ", paths_rel1_rd)
            # print("paths rel2 rd are loop1 ", paths_rel2_rd)
            output = quesans._predict_corechain(question,paths,paths_rel1_rd,paths_rel2_rd)
        elif model == 'slotptr' or model == 'bert_slotptr':
            paths_rel1_sp, paths_rel2_sp, _, _ = create_rd_sp_paths(paths)
            output= quesans._predict_corechain(question,paths,paths_rel1_sp,paths_rel2_sp)
        elif model == 'slotptr_randomvec':
            paths_rel1_sp, paths_rel2_sp, paths_rel1_rd, paths_rel2_rd = create_rd_sp_paths(paths)
            output = quesans._predict_corechain(_q=question, _p=paths, _p1=paths_rel1_sp, _p2=paths_rel2_sp,
                                                _p1_randomvec=paths_rel1_rd, _p2_randomvec=paths_rel2_rd)
        else:
            output = quesans._predict_corechain(question, paths)
        best_path_index = np.argmax(output)
        best_path = paths[best_path_index]

    elif not no_positive_path and len(negative_paths) != 0:
        '''
            There exists positive path and also negative paths
            path = positive_path + negative_paths    
        '''
        if model == 'reldet':

            _, _, paths_rel1_rd, paths_rel2_rd = create_rd_sp_paths(paths)
            # print("paths rel1 rd are loop1 ", paths_rel1_rd)
            # print("paths rel2 rd are loop1 ", paths_rel2_rd)

            output = quesans._predict_corechain(question,paths,paths_rel1_rd,paths_rel2_rd)
        elif model == 'slotptr' or model == 'bert_slotptr':
            paths_rel1_sp, paths_rel2_sp, _, _ = create_rd_sp_paths(paths)
            # print("paths rel1 rd are loop1 ", paths_rel1_sp)
            # print("paths rel2 rd are loop1 ", paths_rel2_sp)
            output= quesans._predict_corechain(question,paths,paths_rel1_sp,paths_rel2_sp)

        elif model == 'slotptr_randomvec':
            paths_rel1_sp, paths_rel2_sp, paths_rel1_rd, paths_rel2_rd = create_rd_sp_paths(paths)
            output = quesans._predict_corechain(_q=question, _p=paths, _p1 = paths_rel1_sp, _p2 = paths_rel2_sp,
            _p1_randomvec = paths_rel1_rd, _p2_randomvec = paths_rel2_rd)

        else:
            output = quesans._predict_corechain(question, paths)
        best_path_index = np.argmax(output)
        best_path = paths[best_path_index]

        # Calculate mrr here
        mrr = 0
        if best_path_index == 0:
            path_predicted_correct = True

        mrr_output = np.argsort(output)[::-1]
        mrr_output = mrr_output.tolist()
        mrr = mrr_output.index(0) + 1.0

        print(output)
        if mrr != 0:
            mrr = 1.0 / mrr

    else:
        print("The code should not have been herr. There is no warning. RUN!!!!!!!!")
        raise ValueError

    return mrr, best_path, path_predicted_correct


def answer_question(qa, index, data, relations, parameter_dict):
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
                                                                                   relations=relations)

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
                                                     negative_paths, no_positive_path,parameter_dict['corechainmodel'],qa)

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

    # rdftype = "none"
    if rdftype == "none":

        pass

    else:
        """
            We do need an rdf constraint.
            We let the rdf class model (ranker) choose between both x and uri paths, 
                and the rdf type model is just used to see if we need paths at all.
        """
        rdf_candidates = sparql_constructor.rdf_type_candidates(data, best_path,
                                                                relations)

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


def sparql_answer(sparql,dbi=None):
    if not dbi:
        dbi = dbp
    test_answer = []
    interface_test_answer = dbi.get_answer(sparql)
    for key in interface_test_answer:
        test_answer = test_answer + interface_test_answer[key]
    return list(set(test_answer))


def _evaluate_sparqls_(test_sparql, true_sparql, type, ground_type,dbp):
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
        test_answer = sparql_answer(test_sparql,dbp)
        true_answer = sparql_answer(true_sparql,dbp)
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
        count_test = sparql_answer(test_sparql,dbp)
        count_true = sparql_answer(true_sparql,dbp)
        if count_test == count_true:
            return 1.0 ,1.0 ,1.0
        else:
            return 0.0 ,0.0 ,1.0

    if type == "ask":
        if dbp.get_answer(test_sparql) == dbp.get_answer(true_sparql):
            return 1.0 ,1.0 ,1.0
        else:
            return 0.0 ,0.0 ,1.0


def evaluate(_logging,dbp):
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
        try:
            f, p, r = _evaluate_sparqls_(test_sparql=log['pred_sparql'],
                                         true_sparql=log['true_sparql'],
                                         type=log['log']['pred_intent'],
                                         ground_type=log['log']['true_intent'],dbp=dbp)
        except:
            print("traced backed")
            traceback.print_exc()
            raise IOError
            f,p,r = 0.0,0.0,0.0
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

    _logging['corechain_accuracy'] = np.mean(corechain_acc)
    _logging['corechain_mean_rr'] = np.mean(corechain_mrr)
    _logging['intent_accuracy'] = np.mean(intent_acc)
    _logging['rdf_type_accuracy'] = np.mean(rdf_type_acc)
    _logging['precision'] = np.mean(overall_p)
    _logging['recall'] = np.mean(overall_r)
    _logging['f1'] = ((2.0*_logging['precision']*_logging['recall'])/(_logging['precision']+_logging['recall']))
    return _logging



if __name__ == "__main__":

    device = torch.device("cuda")
    # sparql_constructor.init(embeddings_interface)
    dbp = db_interface.DBPedia(_verbose=True, caching=True)
    vocabularize_relation = lambda path: embeddings_interface.vocabularize(
        nlutils.tokenize(dbp.get_label(path))).tolist()

    # Reading and setting up config parser
    config = ConfigParser.ConfigParser()
    config.readfp(open('configs/macros.cfg'))

    # setting up device,model name and loss types.
    training_model = 'slotptr_randomvec'
    _dataset = 'lcquad'
    pointwise = False

    # 19 is performing the best
    training_model_number = 7
    _debug = False

    # Loading relations file.
    COMMON_DATA_DIR = 'data/data/common'
    INTENTS = ['count', 'ask', 'list']
    RDFTYPES = ['x', 'uri', 'none']

    _dataset_specific_data_dir = 'data/data/%(dataset)s/' % {'dataset': _dataset}
    _inv_relations = aux.load_inverse_relation(COMMON_DATA_DIR)
    _word_to_id = aux.load_word_list(COMMON_DATA_DIR)
    glove_id_sf_to_glove_id_rel = dl.create_relation_lookup_table(COMMON_DATA_DIR)

    # Model specific paramters
    # #Model specific paramters
    if pointwise:
        training_config = 'pointwise'
    else:
        training_config = 'pairwise'

    parameter_dict = cl.runtime_parameters(dataset=_dataset, training_model=training_model,
                                           training_config=training_config, config_file='configs/macros.cfg')

    if training_model == 'cnn_dot':
        parameter_dict['output_dim'] = int(config.get(_dataset, 'output_dim'))

    # Update parameters
    parameter_dict['_dataset_specific_data_dir'] = _dataset_specific_data_dir
    parameter_dict['_model_dir'] = './data/models/'

    parameter_dict['corechainmodel'] = training_model
    parameter_dict['corechainmodelnumber'] = str(training_model_number)

    parameter_dict['intentmodel'] = 'bilstm_dense'
    parameter_dict['intentmodelnumber'] = '0'

    parameter_dict['rdftypemodel'] = 'bilstm_dense'
    parameter_dict['rdftypemodelnumber'] = '0'

    parameter_dict['rdfclassmodel'] = 'bilstm_dot'
    parameter_dict['rdfclassmodelnumber'] = '0'

    print(parameter_dict)

    parameter_dict['vocab'] = pickle.load(open('resources/vocab_gl.pickle', 'rb'))

    TEMP = aux.data_loading_parameters(_dataset, parameter_dict, runtime=True)

    _dataset_specific_data_dir, _model_specific_data_dir, _file, \
    _max_sequence_length, _neg_paths_per_epoch_train, \
    _neg_paths_per_epoch_validation, _training_split, _validation_split, _index = TEMP

    _data,  _vectors = dl.create_dataset_runtime(file=_file, _dataset=_dataset,
                                                                         _dataset_specific_data_dir=_dataset_specific_data_dir,
                                                                         split_point=.80)

    parameter_dict['vectors'] = _vectors

    # For interpretability's sake
    word_to_gloveid, gloveid_to_word = aux.load_embeddingid_gloveid()

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

    quesans = QuestionAnswering(parameter_dict, pointwise, _word_to_id, device,_dataset, False)

    # Some logs which run during runtime, not after.
    core_chain_acc_log = []
    core_chain_mrr_log = []

    startindex = 0
    for index, data in enumerate(_data[startindex:]):
        print (index)
        if index == 16 or index == 25:
            continue
        print(data.keys())
        index += startindex

        log, metrics = answer_question(qa=quesans,
                                       index=index,
                                       data=data,
                                       relations=_inv_relations,
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
                               relations=_inv_relations)
        # sparql = ""
        # metrics = eval(data, log, metrics)

        # Update logs
        Logging['runtime'].append({'log': log, 'metrics': metrics,
                                   'pred_sparql': sparql, 'true_sparql': data['parsed-data']['node']['sparql_query']})

        # Update metrics
        core_chain_acc_log.append(metrics['core_chain_accuracy_counter'])
        core_chain_mrr_log.append(metrics['core_chain_mrr_counter'])

        # Make shit interpretable
        question = aux.id_to_word(log['question'], gloveid_to_word, remove_pad=True)
        true_path = aux.id_to_word(log['true_path'], gloveid_to_word, remove_pad=True)
        pred_path = aux.id_to_word(log['pred_path'], gloveid_to_word, remove_pad=True)

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

    Logging = evaluate(Logging,dbp)

    model_path = os.path.join(parameter_dict['_model_dir'], 'core_chain')
    model_path = os.path.join(model_path, parameter_dict['corechainmodel'])
    model_path = os.path.join(model_path, parameter_dict['dataset'])
    model_path = os.path.join(model_path, parameter_dict['corechainmodelnumber'])
    pickle.dump(Logging,open(model_path+'/result.pickle','wb+'))

