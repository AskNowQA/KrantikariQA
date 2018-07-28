'''
    Calls various components files to create a models/network. Also provides various functionalities to train and predict.
'''

import components as com
import torch
from qelos_core.scripts.lcquad.corerank import FlatEncoder



class BiLstmDot:

    def __init__(self, _parameter_dict, _word_to_id, _device, _pointwise=False, _debug=False):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise
        self.word_to_id = _word_to_id

        if self.debug:
            print("Init Models")

        self.encoder = FlatEncoder(embdim=self.parameter_dict['embedding_dim'],
                             dims=[self.parameter_dict['hidden_size']],
                             word_dic=self.word_to_id,
                             bidir=True).to(self.device)

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
            question = self.encoder(ques.long())
            paths = self.encoder(paths.long())
            score = torch.sum(question * paths, -1)
            return score

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.

        :return: [(key, model)]
        """
        return [('encoder', self.encoder)]


class BiLstmDense:
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

        self.encoder = FlatEncoder(embdim=self.parameter_dict['embedding_dim'],
                                   dims=[self.parameter_dict['hidden_size']],
                                   word_dic=self.word_to_id,
                                   bidir=True).to(self.device)

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
        ques_batch = self.encoder(ques_batch)
        pos_batch = self.encoder(pos_batch)
        neg_batch = self.encoder(neg_batch)

        # Calculating dot score
        pos_scores = self.dense(torch.cat((ques_batch, pos_batch), dim=1))
        neg_scores = self.dense(torch.cat((ques_batch, neg_batch), dim=1))

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
        score = self.dense(torch.cat((ques_batch, pos_batch), dim=1))

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

            question = self.encoder(ques.long())
            paths = self.encoder(paths.long())
            score = self.dense(torch.cat((question, paths), dim=1))

            return score

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.

        :return: [(key, model)]
        """
        return [('encoder', self.encoder), ('dense', self.dense)]


class BiLstmDenseDot:
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
        self.encoder = FlatEncoder(embdim=self.parameter_dict['embedding_dim'],
                                   dims=[self.parameter_dict['hidden_size']],
                                   word_dic=self.word_to_id,
                                   bidir=True).to(self.device)
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
        ques_batch = self.encoder(ques_batch)
        pos_batch = self.encoder(pos_batch)
        neg_batch = self.encoder(neg_batch)

        # Pass all encoded stuff through the dense too
        ques_batch = self.dense(ques_batch)
        pos_batch = self.dense(pos_batch)
        neg_batch = self.dense(neg_batch)

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

            question = self.encoder(ques.long())
            paths = self.encoder(paths.long())

            # Dense
            question = self.dense(question)
            paths = self.dense(paths)

            score = torch.sum(question * paths, -1)

            return score

    def prepare_save(self):
        """

            This function is called when someone wants to save the underlying models.
            Returns a tuple of key:model pairs which is to be interpreted within save model.

        :return: [(key, model)]
        """
        return [('encoder', self.encoder), ('dense', self.dense)]
