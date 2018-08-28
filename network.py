'''
    Calls various components files to create a models/network. Also provides various functionalities to train and predict.
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from qelos_core.scripts.lcquad.corerank import FlatEncoder

# Local imports
import components as com
from utils import tensor_utils as tu

class Model(object):
    """
        Boilerplate class which helps others have some common functionality.
        These are made with some debugging/loading and with corechains in mind

    """

    def prepare_save(self):
        pass

    def load_from(self, location):
        # Pull the data from disk
        model_dump = torch.load(location)

        # Load parameters
        for key in self.prepare_save():
            key[1].load_state_dict(model_dump[key[0]])

    def get_parameter_sum(self):

        sum = 0
        for model in self.prepare_save():

            model_sum = 0
            for x in list(model[1].parameters()):

                model_sum += np.sum(x.data.cpu().numpy().flatten())

            sum += model_sum

        return sum


class BiLstmDotOld(Model):

    def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):

        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id

        if self.debug:
            print("Init Models")

        self.encoder = com.Encoder(self.parameter_dict['max_length'], self.parameter_dict['hidden_size']
                                   , self.parameter_dict['number_of_layer'], self.parameter_dict['embedding_dim'],
                                   self.parameter_dict['vocab_size'],
                                   bidirectional=self.parameter_dict['bidirectional'],
                                   vectors=self.parameter_dict['vectors']).cuda(self.device)

    def train(self, data, optimizer, loss_fn, device):
        if self.pointwise:
            return self._train_pointwise_(data, optimizer, loss_fn, device)
        else:
            return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object

            returns loss
        '''

        # Unpacking the data and model from args
        ques_batch, pos_batch, neg_batch, y_label = data['ques_batch'], data['pos_batch'], data['neg_batch'], data['y_label']
        hidden = self.encoder.init_hidden(ques_batch.shape[0], device)


        optimizer.zero_grad()
        #Encoding all the data
        ques_batch, _ = self.encoder(ques_batch, hidden)
        pos_batch, _ = self.encoder(pos_batch, hidden)
        neg_batch, _ = self.encoder(neg_batch, hidden)
        # Calculating dot score
        pos_scores = torch.sum(ques_batch * pos_batch, -1)
        neg_scores = torch.sum(ques_batch * neg_batch, -1)
        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''
        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        optimizer.step()
        return loss

    def _train_pointwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, paths and y labels}
            :params models list of [models]
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object

            returrns loss
        '''
        # Unpacking the data and model from args
        ques_batch, path_batch, y_label = data['ques_batch'], data['path_batch'], data['y_label']

        optimizer.zero_grad()

        # Encoding all the data
        ques_batch = self.encoder(ques_batch)
        pos_batch = self.encoder(path_batch)

        # Calculating dot score
        score = torch.sum(ques_batch* pos_batch, -1)

        '''
            Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
        '''
        loss = loss_fn(score, y_label)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, ques, paths, device):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():

            self.encoder.eval()

            hidden = self.encoder.init_hidden(ques.shape[0], device)
            question, _ = self.encoder(ques.long(), hidden)
            paths, _ = self.encoder(paths.long(), hidden)
            score = torch.sum(question[-1] * paths[-1], -1)

            self.encoder.train()

            return score

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.

        :return: [(key, model)]
        """
        return [('encoder', self.encoder)]


# class BiLstmDot(Model):
#
#     def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):
#
#         self.debug = _debug
#         self.parameter_dict = _parameter_dict
#         self.device = _device
#         self.pointwise = _pointwise
#         self.word_to_id = _word_to_id
#
#         if self.debug:
#             print("Init Models")
#
#         self.encoder = FlatEncoder(embdim=self.parameter_dict['embedding_dim'],
#                                    dims=[self.parameter_dict['hidden_size']],
#                                    word_dic=self.word_to_id,
#                                    bidir=True,dropout_rec=self.parameter_dict['dropout_rec'],
#                                    dropout_in=self.parameter_dict['dropout_in']).to(self.device)
#
#     def train(self, data, optimizer, loss_fn, device):
#     #
#         if self.pointwise:
#             return self._train_pointwise_(data, optimizer, loss_fn, device)
#         else:
#             return self._train_pairwise_(data, optimizer, loss_fn, device)
#
#     def _train_pairwise_(self, data, optimizer, loss_fn, device):
#         '''
#             Given data, passes it through model, inited in constructor, returns loss and updates the weight
#             :params data: {batch of question, pos paths, neg paths and dummy y labels}
#             :params optimizer: torch.optim object
#             :params loss fn: torch.nn loss object
#             :params device: torch.device object
#
#             returns loss
#         '''
#
#         # Unpacking the data and model from args
#         ques_batch, pos_batch, neg_batch, y_label = data['ques_batch'], data['pos_batch'], data['neg_batch'], data['y_label']
#
#         optimizer.zero_grad()
#         #Encoding all the data
#
#
#
#         ques_batch = self.encoder(ques_batch)
#         pos_batch = self.encoder(pos_batch)
#         neg_batch = self.encoder(neg_batch)
#
#
#
#         #Calculating dot score
#         pos_scores = torch.sum(ques_batch * pos_batch, -1)
#         neg_scores = torch.sum(ques_batch * neg_batch, -1)
#         '''
#             If `y == 1` then it assumed the first input should be ranked higher
#             (have a larger value) than the second input, and vice-versa for `y == -1`
#         '''
#         loss = loss_fn(pos_scores, neg_scores, y_label)
#         loss.backward()
#         optimizer.step()
#         return loss
#
#     def _train_pointwise_(self, data, optimizer, loss_fn, device):
#         '''
#             Given data, passes it through model, inited in constructor, returns loss and updates the weight
#             :params data: {batch of question, paths and y labels}
#             :params models list of [models]
#             :params optimizer: torch.optim object
#             :params loss fn: torch.nn loss object
#             :params device: torch.device object
#             returrns loss
#         '''
#         # Unpacking the data and model from args
#         ques_batch, path_batch, y_label = data['ques_batch'], data['path_batch'], data['y_label']
#
#         optimizer.zero_grad()
#
#         # Encoding all the data
#         ques_batch = self.encoder(ques_batch)
#         pos_batch = self.encoder(path_batch)
#
#         #
#         # norm_ques_batch = torch.abs(torch.norm(ques_batch,dim=1,p=1))
#         # norm_pos_batch = torch.abs(torch.norm(pos_batch,dim=1,p=1))
#
#         # ques_batch = F.normalize(F.relu(ques_batch),p=1,dim=1)
#         # pos_batch = F.normalize(F.relu(pos_batch),p=1,dim=1)
#         # ques_batch =(F.normalize(ques_batch,p=1,dim=1)/2) + .5
#         # pos_batch =(F.normalize(pos_batch,p=1,dim=1)/2) + .5
#
#
#
#
#         # Calculating dot score
#         score = torch.sum(ques_batch * pos_batch, -1)
#         # score = score.div(norm_ques_batch*norm_pos_batch).div_(2.0).add_(0.5)
#             # print("shape of score is,", score.shape)
#             # print("score is , ", score)
#             #
#             #
#             # print("shape of y label is ", y_label.shape)
#             # print("value of y label is ", y_label)
#
#         # raise ValueError
#
#         '''
#             Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
#         '''
#         loss = loss_fn(score, y_label)
#         loss.backward()
#         optimizer.step()
#
#         return loss
#
#     def predict(self, ques, paths, device):
#         """
#             Same code works for both pairwise or pointwise
#         """
#         with torch.no_grad():
#
#             self.encoder.eval()
#             question = self.encoder(ques.long())
#             paths = self.encoder(paths.long())
#             if self.pointwise:
#                 # question = F.normalize(F.relu(question),p=1,dim=1)
#                 # paths = F.normalize(F.relu(paths),p=1,dim=1)
#                 # norm_ques_batch = torch.abs(torch.norm(question, dim=1, p=1))
#                 # norm_pos_batch = torch.abs(torch.norm(paths, dim=1, p=1))
#                 score = torch.sum(question * paths, -1)
#                 # score = score.div(norm_ques_batch * norm_pos_batch).div_(2.0).add_(0.5)
#             else:
#                 score = torch.sum(question * paths, -1)
#
#             self.encoder.train()
#             return score
#
#     def prepare_save(self):
#         """
#
#             This function is called when someone wants to save the underlying models.
#             Returns a tuple of key:model pairs which is to be interpreted within save model.
#
#         :return: [(key, model)]
#         """
#         return [('encoder', self.encoder)]
#
#     def load_from(self, location):
#         # Pull the data from disk
#         if self.debug: print("loading Bilstmdot model from", location)
#         self.encoder.load_state_dict(torch.load(location)['encoder'])
#         if self.debug: print("model loaded with weights ,", self.get_parameter_sum())
#
#         # # Load parameters
#         # for key in self.prepare_save():
#         #     key[1].load_state_dict(model_dump[key[0]])


class BiLstmDot(Model):

    def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):

        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id

        if self.debug:
            print("Init Models")



        self.encoder = com.BetterEncoder(max_length = self.parameter_dict['max_length'],
                                         hidden_dim = self.parameter_dict['hidden_size'],
                                         number_of_layer = self.parameter_dict['number_of_layer'],
                                         embedding_dim = self.parameter_dict['embedding_dim'],
                                         vocab_size = self.parameter_dict['vocab_size'], bidirectional = True,
                                         dropout = self.parameter_dict['dropout'], mode = 'LSTM', enable_layer_norm = False,
                                         vectors = self.parameter_dict['vectors'], debug = self.debug).to(self.device)

    def train(self, data, optimizer, loss_fn, device):
    #
        if self.pointwise:
            return self._train_pointwise_(data, optimizer, loss_fn, device)
        else:
            return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object

            returns loss
        '''

        # Unpacking the data and model from args
        ques_batch, pos_batch, neg_batch, y_label = data['ques_batch'], data['pos_batch'], data['neg_batch'], data['y_label']

        optimizer.zero_grad()
        #Encoding all the data

        hidden = self.encoder.init_hidden(ques_batch.shape[0],self.device)
        _, ques_batch_encoded, _, _ = self.encoder(tu.trim(ques_batch), hidden)
        _, pos_batch_encoded, _, _ = self.encoder(tu.trim(pos_batch), hidden)
        _, neg_batch_encoded, _, _  = self.encoder(tu.trim(neg_batch), hidden)



        #Calculating dot score
        pos_scores = torch.sum(ques_batch_encoded * pos_batch_encoded, -1)
        neg_scores = torch.sum(ques_batch_encoded * neg_batch_encoded, -1)
        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''
        try:
            loss = loss_fn(pos_scores, neg_scores, y_label)
        except RuntimeError:
            print(pos_scores.shape, neg_scores.shape, y_label.shape,  ques_batch.shape, pos_batch.shape, neg_batch.shape)
        loss.backward()
        optimizer.step()
        return loss

    def _train_pointwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, paths and y labels}
            :params models list of [models]
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object
            returrns loss
        '''
        # Unpacking the data and model from args
        ques_batch, path_batch, y_label = data['ques_batch'], data['path_batch'], data['y_label']

        optimizer.zero_grad()

        # Encoding all the data
        hidden = self.encoder.init_hidden(ques_batch.shape[0], self.device)
        _, ques_batch, _, _ = self.encoder(tu.trim(ques_batch), hidden)
        _, pos_batch, _, _ = self.encoder(tu.trim(path_batch), hidden)

        #
        # norm_ques_batch = torch.abs(torch.norm(ques_batch,dim=1,p=1))
        # norm_pos_batch = torch.abs(torch.norm(pos_batch,dim=1,p=1))

        # ques_batch = F.normalize(F.relu(ques_batch),p=1,dim=1)
        # pos_batch = F.normalize(F.relu(pos_batch),p=1,dim=1)
        # ques_batch =(F.normalize(ques_batch,p=1,dim=1)/2) + .5
        # pos_batch =(F.normalize(pos_batch,p=1,dim=1)/2) + .5




        # Calculating dot score
        score = torch.sum(ques_batch * pos_batch, -1)
        # score = score.div(norm_ques_batch*norm_pos_batch).div_(2.0).add_(0.5)
            # print("shape of score is,", score.shape)
            # print("score is , ", score)
            #
            #
            # print("shape of y label is ", y_label.shape)
            # print("value of y label is ", y_label)

        # raise ValueError

        '''
            Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
        '''
        loss = loss_fn(score, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def predict(self, ques, paths, device):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():

            self.encoder.eval()
            hidden = self.encoder.init_hidden(ques.shape[0], self.device)

            _, question, _, _ = self.encoder(tu.trim(ques.long()), hidden)
            _, paths, _, _ = self.encoder(tu.trim(paths.long()), hidden)

            if self.pointwise:
                # question = F.normalize(F.relu(question),p=1,dim=1)
                # paths = F.normalize(F.relu(paths),p=1,dim=1)
                # norm_ques_batch = torch.abs(torch.norm(question, dim=1, p=1))
                # norm_pos_batch = torch.abs(torch.norm(paths, dim=1, p=1))
                score = torch.sum(question * paths, -1)
                # score = score.div(norm_ques_batch * norm_pos_batch).div_(2.0).add_(0.5)
            else:
                score = torch.sum(question * paths, -1)

            self.encoder.train()
            return score

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.

        :return: [(key, model)]
        """
        return [('encoder', self.encoder)]

    def load_from(self, location):
        # Pull the data from disk
        if self.debug: print("loading Bilstmdot model from", location)
        self.encoder.load_state_dict(torch.load(location)['encoder'])
        if self.debug: print("model loaded with weights ,", self.get_parameter_sum())


class BiLstmDot_ULMFiT(Model):

    def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):

        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id

        # Load the pre-trained model
        pretrained_weights = torch.load('./ulmfit/wt103/fwd_wt103_enc.h5', map_location= lambda storage, loc: storage)
        new_vectors = self.parameter_dict['vectors']


        if self.debug:
            print("Init Models")

        self.encoder = FlatEncoder(embdim=self.parameter_dict['embedding_dim'],
                                   dims=[self.parameter_dict['hidden_size']],
                                   word_dic=self.word_to_id,
                                   bidir=True,dropout_rec=self.parameter_dict['dropout_rec'],
                                   dropout_in=self.parameter_dict['dropout_in']).to(self.device)

    def train(self, data, optimizer, loss_fn, device):
    #
        if self.pointwise:
            return self._train_pointwise_(data, optimizer, loss_fn, device)
        else:
            return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object

            returns loss
        '''

        # Unpacking the data and model from args
        ques_batch, pos_batch, neg_batch, y_label = data['ques_batch'], data['pos_batch'], data['neg_batch'], data['y_label']

        optimizer.zero_grad()
        #Encoding all the data



        ques_batch = self.encoder(ques_batch)
        pos_batch = self.encoder(pos_batch)
        neg_batch = self.encoder(neg_batch)



        #Calculating dot score
        pos_scores = torch.sum(ques_batch * pos_batch, -1)
        neg_scores = torch.sum(ques_batch * neg_batch, -1)
        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''
        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        optimizer.step()
        return loss

    def _train_pointwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, paths and y labels}
            :params models list of [models]
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object
            returrns loss
        '''
        # Unpacking the data and model from args
        ques_batch, path_batch, y_label = data['ques_batch'], data['path_batch'], data['y_label']

        optimizer.zero_grad()

        # Encoding all the data
        ques_batch = self.encoder(ques_batch)
        pos_batch = self.encoder(path_batch)

        #
        # norm_ques_batch = torch.abs(torch.norm(ques_batch,dim=1,p=1))
        # norm_pos_batch = torch.abs(torch.norm(pos_batch,dim=1,p=1))

        # ques_batch = F.normalize(F.relu(ques_batch),p=1,dim=1)
        # pos_batch = F.normalize(F.relu(pos_batch),p=1,dim=1)
        # ques_batch =(F.normalize(ques_batch,p=1,dim=1)/2) + .5
        # pos_batch =(F.normalize(pos_batch,p=1,dim=1)/2) + .5




        # Calculating dot score
        score = torch.sum(ques_batch * pos_batch, -1)
        # score = score.div(norm_ques_batch*norm_pos_batch).div_(2.0).add_(0.5)
            # print("shape of score is,", score.shape)
            # print("score is , ", score)
            #
            #
            # print("shape of y label is ", y_label.shape)
            # print("value of y label is ", y_label)

        # raise ValueError

        '''
            Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
        '''
        loss = loss_fn(score, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def predict(self, ques, paths, device):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():

            self.encoder.eval()
            question = self.encoder(ques.long())
            paths = self.encoder(paths.long())
            if self.pointwise:
                # question = F.normalize(F.relu(question),p=1,dim=1)
                # paths = F.normalize(F.relu(paths),p=1,dim=1)
                # norm_ques_batch = torch.abs(torch.norm(question, dim=1, p=1))
                # norm_pos_batch = torch.abs(torch.norm(paths, dim=1, p=1))
                score = torch.sum(question * paths, -1)
                # score = score.div(norm_ques_batch * norm_pos_batch).div_(2.0).add_(0.5)
            else:
                score = torch.sum(question * paths, -1)

            self.encoder.train()
            return score

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.

        :return: [(key, model)]
        """
        return [('encoder', self.encoder)]

    def load_from(self, location):
        # Pull the data from disk
        if self.debug: print("loading Bilstmdot model from", location)
        self.encoder.load_state_dict(torch.load(location)['encoder'])
        if self.debug: print("model loaded with weights ,", self.get_parameter_sum())

        # # Load parameters
        # for key in self.prepare_save():
        #     key[1].load_state_dict(model_dump[key[0]])


class BiLstmDense(Model):
    """
        This model replaces the dot product of BiLstmDot with a two layered dense classifier.
        Plus, we only use the last state of the encoder in doing so.
        Like before, we have both pairwise and pointwise versions.
    """

    def __init__(self, _parameter_dict, _word_to_id,  _device, _pointwise=False, _debug=False):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id
        self.hiddendim = self.parameter_dict['hidden_size'] * (2 * int(self.parameter_dict['bidirectional']))

        if self.debug:
            print("Init Models")

        self.encoder = com.BetterEncoder(max_length=self.parameter_dict['max_length'],
                                         hidden_dim=self.parameter_dict['hidden_size'],
                                         number_of_layer=self.parameter_dict['number_of_layer'],
                                         embedding_dim=self.parameter_dict['embedding_dim'],
                                         vocab_size=self.parameter_dict['vocab_size'], bidirectional=True,
                                         dropout=self.parameter_dict['dropout'], mode='LSTM', enable_layer_norm=False,
                                         vectors=self.parameter_dict['vectors'], debug=self.debug).to(self.device)

        self.dense = com.DenseClf(inputdim=self.hiddendim*2,        # *2 because we have two things concatinated here
                                  hiddendim=self.hiddendim/2,
                                  outputdim=1).to(self.device)

    def train(self, data, optimizer, loss_fn, device):

        if self.pointwise:
            return self._train_pointwise_(data, optimizer, loss_fn, device)
        else:
            return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object

            returns loss
        '''

        # Unpacking the data and model from args
        ques_batch, pos_batch, neg_batch, y_label = data['ques_batch'], data['pos_batch'], data['neg_batch'], data[
            'y_label']

        optimizer.zero_grad()
        # Encoding all the data
        hidden = self.encoder.init_hidden(ques_batch.shape[0], self.device)
        _, ques_batch_encoded, _, _ = self.encoder(tu.trim(ques_batch), hidden)
        _, pos_batch_encoded, _, _ = self.encoder(tu.trim(pos_batch), hidden)
        _, neg_batch_encoded, _, _  = self.encoder(tu.trim(neg_batch), hidden)

        # Calculating dot score
        pos_scores = self.dense(torch.cat((ques_batch_encoded, pos_batch_encoded), dim=1))
        neg_scores = self.dense(torch.cat((ques_batch_encoded, neg_batch_encoded), dim=1))

        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''

        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def _train_pointwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, paths and y labels}
            :params models list of [models]
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object

            returrns loss
        '''
        # Unpacking the data and model from args
        ques_batch, path_batch, y_label = data['ques_batch'], data['path_batch'], data['y_label']

        optimizer.zero_grad()

        # Encoding all the data
        hidden = self.encoder.init_hidden(ques_batch.shape[0], self.device)
        _, ques_batch, _, _ = self.encoder(tu.trim(ques_batch), hidden)
        _, pos_batch, _, _ = self.encoder(tu.trim(path_batch), hidden)

        # Calculating dot score
        score = self.dense(torch.cat((ques_batch, pos_batch), dim=1)).squeeze()

        '''
            Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
        '''

        loss = loss_fn(score, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def predict(self, ques, paths, device):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():

            self.encoder.eval()
            hidden = self.encoder.init_hidden(ques.shape[0], self.device)

            _, question, _, _ = self.encoder(tu.trim(ques.long()), hidden)
            _, paths, _, _ = self.encoder(tu.trim(paths.long()), hidden)
            score = self.dense(torch.cat((question, paths), dim=1)).squeeze()
            self.encoder.train()

            return score

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.

        :return: [(key, model)]
        """
        return [('encoder', self.encoder), ('dense', self.dense)]


class BiLstmDenseDot(Model):
    """
        This model uses the encoder, then condenses the vector into something of a smaller dimension.
        Then uses a regular dot product to compute the final deal.
        Plus, we only use the last state of the encoder in doing so.
        Like before, we have both pairwise and pointwise versions.
    """

    def __init__(self, _parameter_dict, _word_to_id,  _device, _pointwise=False, _debug=False):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.word_to_id = _word_to_id
        self.pointwise = _pointwise

        self.hiddendim = self.parameter_dict['hidden_size'] * (2 * int(self.parameter_dict['bidirectional']))

        if self.debug:
            print("Init Models")
        self.encoder = com.BetterEncoder(max_length=self.parameter_dict['max_length'],
                                         hidden_dim=self.parameter_dict['hidden_size'],
                                         number_of_layer=self.parameter_dict['number_of_layer'],
                                         embedding_dim=self.parameter_dict['embedding_dim'],
                                         vocab_size=self.parameter_dict['vocab_size'], bidirectional=True,
                                         dropout=self.parameter_dict['dropout'], mode='LSTM', enable_layer_norm=False,
                                         vectors=self.parameter_dict['vectors'], debug=self.debug).to(self.device)

        self.dense = com.DenseClf(inputdim=self.hiddendim,
                                  hiddendim=self.hiddendim/2,
                                  outputdim=self.hiddendim/4).to(self.device)

    def train(self, data, optimizer, loss_fn, device):

        if self.pointwise:
            return self._train_pointwise_(data, optimizer, loss_fn, device)
        else:
            return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object

            returns loss
        '''

        # Unpacking the data and model from args
        ques_batch, pos_batch, neg_batch, y_label = data['ques_batch'], data['pos_batch'], data['neg_batch'], data[
            'y_label']

        optimizer.zero_grad()

        # Encoding all the data
        hidden = self.encoder.init_hidden(ques_batch.shape[0], self.device)
        _, ques_batch_encoded, _, _ = self.encoder(tu.trim(ques_batch), hidden)
        _, pos_batch_encoded, _, _ = self.encoder(tu.trim(pos_batch), hidden)
        _, neg_batch_encoded, _, _ = self.encoder(tu.trim(neg_batch), hidden)


        # Pass all encoded stuff through the dense too
        ques_batch = self.dense(ques_batch_encoded)
        pos_batch = self.dense(pos_batch_encoded)
        neg_batch = self.dense(neg_batch_encoded)


        # Calculating dot score
        pos_scores = torch.sum(ques_batch * pos_batch, -1)
        neg_scores = torch.sum(ques_batch * neg_batch, -1)

        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''

        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def _train_pointwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, paths and y labels}
            :params models list of [models]
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object

            returrns loss
        '''
        # Unpacking the data and model from args
        ques_batch, path_batch, y_label = data['ques_batch'], data['path_batch'], data['y_label']

        optimizer.zero_grad()

        # Encoding all the data
        hidden = self.encoder.init_hidden(ques_batch.shape[0], self.device)
        _, ques_batch, _, _ = self.encoder(tu.trim(ques_batch), hidden)
        _, pos_batch, _, _ = self.encoder(tu.trim(path_batch), hidden)

        # Compress the last state
        ques_batch = self.dense(ques_batch)
        pos_batch = self.dense(pos_batch)

        # Calculating dot score
        score = torch.sum(ques_batch * pos_batch, -1)

        '''
            Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
        '''

        loss = loss_fn(score, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def predict(self, ques, paths, device):
        """
            Same code works for both pairwise or pointwise
        """

        with torch.no_grad():

            self.encoder.eval()

            hidden = self.encoder.init_hidden(ques.shape[0], self.device)

            _, question, _, _ = self.encoder(tu.trim(ques.long()), hidden)
            _, paths, _, _ = self.encoder(tu.trim(paths.long()), hidden)

            # Dense
            question = self.dense(question)
            paths = self.dense(paths)

            score = torch.sum(question * paths, -1)

            self.encoder.train()

            return score

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.
https://arxiv.org/pdf/1606.01933.pdf
        :return: [(key, model)]
        """
        return [('encoder', self.encoder), ('dense', self.dense)]


class CNNDot(BiLstmDot):

    def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):

        super(CNNDot, self).__init__( _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False)


        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id

        if self.debug:
            print("Init Models")

        self.encoder = com.CNN(_vectors=self.parameter_dict['vectors'], _vocab_size=self.parameter_dict['vocab_size'] ,
                           _embedding_dim = self.parameter_dict['embedding_dim'] , _output_dim = self.parameter_dict['output_dim'],_debug=self.debug).to(self.device)


class DecomposableAttention(Model):
    """
        Implementation of https://arxiv.org/pdf/1606.01933.pdf.
        Uses an encoder and AttendCompareAggregate class in components
    """
    def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):

        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id
        self.hiddendim = self.parameter_dict['hidden_size'] * (1+ int(self.parameter_dict['bidirectional']))

        if self.debug:
            print("Init Models")

        # self.encoder = FlatEncoder(embdim=self.parameter_dict['embedding_dim'],
        #                            dims=[self.parameter_dict['hidden_size']],
        #                            word_dic=self.word_to_id,
        #                            bidir=True,dropout_rec=self.parameter_dict['dropout_rec'],
        #                            dropout_in=self.parameter_dict['dropout_in']).to(self.device)

        # self.encoder = com.Encoder(self.parameter_dict['max_length'], self.parameter_dict['hidden_size']
        #                            , self.parameter_dict['number_of_layer'], self.parameter_dict['embedding_dim'],
        #                            self.parameter_dict['vocab_size'],
        #                            bidirectional=self.parameter_dict['bidirectional'],
        #                            vectors=self.parameter_dict['vectors']).to(self.device)

        self.encoder = com.BetterEncoder(max_length = self.parameter_dict['max_length'],
                                         hidden_dim = self.parameter_dict['hidden_size'],
                                         number_of_layer = self.parameter_dict['number_of_layer'],
                                         embedding_dim = self.parameter_dict['embedding_dim'],
                                         vocab_size = self.parameter_dict['vocab_size'], bidirectional = True,
                                         dropout = self.parameter_dict['dropout'], mode = 'LSTM', enable_layer_norm = False,
                                         vectors = self.parameter_dict['vectors'], debug = self.debug).to(self.device)

        self.scorer = com.BetterAttendCompareAggregate(inputdim=self.hiddendim, debug=self.debug).to(self.device)

    def train(self, data, optimizer, loss_fn, device):
        if self.pointwise:
            return self._train_pointwise_(data, optimizer, loss_fn, device)
        else:
            return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        ques_batch, pos_batch, neg_batch, y_label = data['ques_batch'], data['pos_batch'], data['neg_batch'], data[
            'y_label']

        hidden = self.encoder.init_hidden(ques_batch.shape[0], device)

        optimizer.zero_grad()
        # Encoding all the data
        ques_batch_encoded, _, _, ques_mask = self.encoder(tu.trim(ques_batch),hidden)
        pos_batch_encoded, _, _, pos_mask = self.encoder(tu.trim(pos_batch), hidden)
        neg_batch_encoded, _, _, neg_mask = self.encoder(tu.trim(neg_batch), hidden)

        # Now, we get a pos and a neg score.
        pos_scores = self.scorer(ques_batch_encoded, pos_batch_encoded, ques_mask, pos_mask)
        neg_scores = self.scorer(ques_batch_encoded, neg_batch_encoded, ques_mask, neg_mask)

        '''
                    If `y == 1` then it assumed the first input should be ranked higher
                    (have a larger value) than the second input, and vice-versa for `y == -1`
        '''

        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        optimizer.step()
        return loss

    def _train_pointwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, paths and y labels}
            :params models list of [models]
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object
            returrns loss
        '''
        # Unpacking the data and model from args
        ques_batch, path_batch, y_label = data['ques_batch'], data['path_batch'], data['y_label']

        hidden = self.encoder.init_hidden(ques_batch.shape[0], device)

        optimizer.zero_grad()

        # Encoding all the data
        ques_batch, _, _, ques_mask = self.encoder(tu.trim(ques_batch), hidden)
        path_batch, _, _, path_mask = self.encoder(tu.trim(path_batch), hidden)

        # Calculating dot score
        score = self.scorer(ques_batch, path_batch, ques_mask, path_mask).squeeze()

        '''
            Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
        '''

        loss = loss_fn(score, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def predict(self, ques, paths, device):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            self.encoder.eval()
            hidden = self.encoder.init_hidden(ques.shape[0], device)

            question, _, _, question_mask = self.encoder(tu.trim(ques.long()),hidden)
            paths, _, _, paths_mask = self.encoder(tu.trim(paths.long()), hidden)
            score = self.scorer(question, paths, question_mask, paths_mask)

            self.encoder.train()
            return score.squeeze()

    def prepare_save(self):
        return [('encoder', self.encoder), ('scorer', self.scorer)]


class RelDetection(Model):

    def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):

        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id
        self.hiddendim = self.parameter_dict['hidden_size'] * (1+ int(self.parameter_dict['bidirectional']))

        if self.debug:
            print("Init Models")

        self.encoder = com.HRBiLSTM(hidden_dim=_parameter_dict['hidden_size'],
                                    max_len_path=_parameter_dict['relrd_pad'],
                                    max_len_ques=_parameter_dict['max_length'],
                                    embedding_dim=_parameter_dict['embedding_dim'],
                                    dropout=_parameter_dict['dropout'],
                                    vocab_size=_parameter_dict['vocab_size'],
                                    vectors=_parameter_dict['vectors'],
                                    debug=self.debug).to(self.device)

    def train(self, data, optimizer, loss_fn, device):
        if self.pointwise:
            return self._train_pointwise_(data, optimizer, loss_fn, device)
        else:
            return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        ques_batch, pos_batch, pos_rel1_batch, pos_rel2_batch, neg_batch, neg_rel1_batch, neg_rel2_batch, y_label = \
            data['ques_batch'], data['pos_batch'], data['pos_rel1_batch'], data['pos_rel2_batch'], \
            data['neg_batch'], data['neg_rel1_batch'], data['neg_rel2_batch'], data['y_label']

        optimizer.zero_grad()

        # Instantiate hidden states
        _hp = self.encoder.init_hidden(ques_batch.shape[0], device=device)
        _hn = self.encoder.init_hidden(ques_batch.shape[0], device=device)

        # Encoding all the data
        pos_scores = self.encoder(ques=ques_batch,
                                  path_word=pos_batch,
                                  path_rel_1=pos_rel1_batch,
                                  path_rel_2=pos_rel2_batch,
                                  _h=_hp)

        neg_scores = self.encoder(ques=ques_batch,
                                  path_word=neg_batch,
                                  path_rel_1=neg_rel1_batch,
                                  path_rel_2=neg_rel2_batch,
                                  _h=_hn)

        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''

        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def _train_pointwise(self, data, optimizer, loss_fn, device):
        ques_batch, path_batch, path_rel1_batch, path_rel2_batch, y_label = \
            data['ques_batch'], data['path_batch'], data['path_rel1_batch'], data['path_rel2_batch'], data['y_label']

        optimizer.zero_grad()

        # Instantiate hidden states
        _h = self.encoder.init_hidden(ques_batch.shape[0], device=device)

        # Encoding all the data
        score = self.encoder(ques=ques_batch,
                             path_word=path_batch,
                             path_rel_1=path_rel1_batch,
                             path_rel_2=path_rel2_batch,
                             _h=_h)

        '''
            Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
        '''
        loss = loss_fn(score, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def predict(self, ques, paths, paths_rel1,paths_rel2, device):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            self.encoder.eval()
            _h = self.encoder.init_hidden(ques.shape[0], device=device)

            score = self.encoder(ques=ques,
                                 path_word=paths,
                                 path_rel_1=paths_rel1,
                                 path_rel_2=paths_rel2,
                                 _h=_h).squeeze()
            self.encoder.train()
            return score

    def prepare_save(self):
        return [('model',self.encoder)]


class SlotPointerModel(Model):

    def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):

        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id

        if self.debug:
            print("Init Models")

        self.encoder_q = com.BetterEncoder(
            number_of_layer=self.parameter_dict['number_of_layer'],
            embedding_dim=self.parameter_dict['embedding_dim'],
            hidden_dim=self.parameter_dict['hidden_size'],
            vocab_size=self.parameter_dict['vocab_size'],
            max_length=self.parameter_dict['max_length'],
            dropout=self.parameter_dict['dropout'],
            vectors=self.parameter_dict['vectors'],
            enable_layer_norm=False,
            bidirectional=True,
            residual=True,
            mode = 'LSTM',
            debug = self.debug).to(self.device)

        self.encoder_p = com.BetterEncoder(
            number_of_layer=self.parameter_dict['number_of_layer'],
            embedding_dim=self.parameter_dict['embedding_dim'],
            hidden_dim=self.parameter_dict['hidden_size'],
            vocab_size=self.parameter_dict['vocab_size'],
            max_length=self.parameter_dict['max_length'],
            dropout=self.parameter_dict['dropout'],
            vectors=self.parameter_dict['vectors'],
            enable_layer_norm=False,
            bidirectional=True,
            residual=True,
            mode = 'LSTM',
            debug = self.debug).to(self.device)

        self.comparer = com.SlotPointer(
            embedding_dim=self.parameter_dict['embedding_dim'],
            max_len_ques=self.parameter_dict['max_length'],
            hidden_dim=self.parameter_dict['hidden_size'],
            max_len_path=self.parameter_dict['relsp_pad'],
            vocab_size=self.parameter_dict['vocab_size'],
            debug=self.debug).to(self.device)

    def train(self, data, optimizer, loss_fn, device):
        if self.pointwise:
            return self._train_pointwise_(data, optimizer, loss_fn, device)
        else:
            return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        ques_batch, pos_1_batch, pos_2_batch, neg_1_batch, neg_2_batch, y_label = \
            data['ques_batch'], data['pos_rel1_batch'], data['pos_rel2_batch'], \
            data['neg_rel1_batch'], data['neg_rel2_batch'], data['y_label']

        hidden = self.encoder_q.init_hidden(ques_batch.shape[0], device)

        optimizer.zero_grad()

        # Have to manually check if the 2nd paths holds anything in this batch.
        # If not, we have to pad everything up with zeros, or even call a limited part of the comparison module.
        pos_2_batch = tu.no_one_left_behind(pos_2_batch)
        neg_2_batch = tu.no_one_left_behind(neg_2_batch)

        # Encoding all the data
        ques_batch_encoded, _, _, ques_mask, ques_batch_embedded, _ = self.encoder_q(tu.trim(ques_batch),hidden)
        _, pos_1_encoded, _, _, _, pos_1_embedded = self.encoder_p(pos_1_batch, hidden)
        _, pos_2_encoded, _, _, _, pos_2_embedded = self.encoder_p(pos_2_batch, hidden)
        _, neg_1_encoded, _, _, _, neg_1_embedded = self.encoder_p(neg_1_batch, hidden)
        _, neg_2_encoded, _, _, _, neg_2_embedded = self.encoder_p(neg_2_batch, hidden)

        # Pass them to the comparison module
        pos_scores = self.comparer(ques_enc=ques_batch_encoded,
                                   ques_emb=ques_batch_embedded,
                                   ques_mask=ques_mask,
                                   path_1_enc=pos_1_encoded,
                                   path_1_emb=pos_1_embedded,
                                   path_2_enc=pos_2_encoded,
                                   path_2_emb=pos_2_embedded)

        # Pass them to the comparison module
        neg_scores = self.comparer(ques_enc=ques_batch_encoded,
                                   ques_emb=ques_batch_embedded,
                                   ques_mask=ques_mask,
                                   path_1_enc=neg_1_encoded,
                                   path_1_emb=neg_1_embedded,
                                   path_2_enc=neg_2_encoded,
                                   path_2_emb=neg_2_embedded)

        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''

        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def _train_pointwise(self, data, optimizer, loss_fn, device):
        ques_batch, path_1_batch, path_2_batch, y_label = \
            data['ques_batch'], data['path_rel1_batch'], data['path_rel2_batch'], data['y_label']

        path_2_batch = tu.no_one_left_behind(path_2_batch)

        hidden = self.encoder_q.init_hidden(ques_batch.shape[0], device)

        optimizer.zero_grad()

        # Encoding all the data
        ques_batch_encoded, _, _, ques_mask, ques_batch_embedded, _ = self.encoder_q(tu.trim(ques_batch), hidden)
        _, pos_1_encoded, _, _, _, pos_1_embedded = self.encoder_p(path_1_batch, hidden)
        _, pos_2_encoded, _, _, _, pos_2_embedded = self.encoder_p(path_2_batch, hidden)

        score = self.comparer(ques_enc=ques_batch_encoded,
                                   ques_emb=ques_batch_embedded,
                                   ques_mask=ques_mask,
                                   path_1_enc=pos_1_encoded,
                                   path_1_emb=pos_1_embedded,
                                   path_2_enc=pos_2_encoded,
                                   path_2_emb=pos_2_embedded)

        '''
            Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
        '''
        loss = loss_fn(score, y_label)
        loss.backward()
        optimizer.step()

        return loss

    def predict(self, ques, paths, paths_rel1,paths_rel2, device):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            self.encoder_q.eval()
            self.encoder_p.eval()
            self.comparer.eval()
            hidden = self.encoder_q.init_hidden(ques.shape[0], device=device)

            paths_rel2 = tu.no_one_left_behind(paths_rel2)

            # Encoding all the data
            ques_batch_encoded, _, _, ques_mask, ques_batch_embedded, _ = self.encoder_q(tu.trim(ques), hidden)
            _, pos_1_encoded, _, _, _, pos_1_embedded = self.encoder_p(paths_rel1, hidden)
            _, pos_2_encoded, _, _, _, pos_2_embedded = self.encoder_p(paths_rel2, hidden)

            score = self.comparer(ques_enc=ques_batch_encoded,
                                  ques_emb=ques_batch_embedded,
                                  ques_mask=ques_mask,
                                  path_1_enc=pos_1_encoded,
                                  path_1_emb=pos_1_embedded,
                                  path_2_enc=pos_2_encoded,
                                  path_2_emb=pos_2_embedded).squeeze()

            self.encoder_q.train()
            self.encoder_p.train()
            self.comparer.train()
            return score

    def prepare_save(self):
        return [('model', self.encoder_q)]