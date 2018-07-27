'''
    Calls various components files to create a models/network. Also provides various functionalities to train and predict.
'''

import components as com
import torch

class BiLstmDot():

    def __init__(self, _parameter_dict, _device, _pointwise=False, _debug=False):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise

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
        #Calculating dot score
        pos_scores = torch.sum(ques_batch[-1] * pos_batch[-1], -1)
        neg_scores = torch.sum(ques_batch[-1] * neg_batch[-1], -1)
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
        ques_batch, _ = self.encoder(ques_batch, hidden)
        pos_batch, _ = self.encoder(path_batch, hidden)

        # Calculating dot score
        score = torch.sum(ques_batch[-1] * pos_batch[-1], -1)

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

        hidden = self.encoder.init_hidden(ques.shape[0], device)
        question, _ = self.encoder(ques.long(), hidden)
        paths, _ = self.encoder(paths.long(), hidden)
        score = torch.sum(question[-1] * paths[-1], -1)
        return score


class BiLstmDense:

    def __init__(self, _parameter_dict, _device, _pointwise=False, _debug=False):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.pointwise = _pointwise

        if self.debug:
            print("Init Models")
        self.encoder = com.Encoder(self.parameter_dict['max_length'], self.parameter_dict['hidden_size'],
                                   self.parameter_dict['number_of_layer'], self.parameter_dict['embedding_dim'],
                                   self.parameter_dict['vocab_size'],
                                   bidirectional=self.parameter_dict['bidirectional'],
                                   vectors=self.parameter_dict['vectors']).cuda(self.device)
        self.dense = com.DenseClf(inputdim=self.parameter_dict['hidden_size']*2,
                                  hiddendim=self.parameter_dict['hidden_size'],
                                  outputdim=1)

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

            hidden = self.encoder.init_hidden(ques_batch.shape[0], device)
            optimizer.zero_grad()
            # Encoding all the data
            ques_batch, _ = self.encoder(ques_batch, hidden)
            pos_batch, _ = self.encoder(pos_batch, hidden)
            neg_batch, _ = self.encoder(neg_batch, hidden)
            # Calculating dot score
            pos_scores = self.dense(torch.cat((ques_batch[-1], pos_batch[-1]), dim=1))
            neg_scores = self.dense(torch.cat((ques_batch[-1], neg_batch[-1]), dim=1))
            #     torch.sum(ques_batch[-1] * pos_batch[-1], -1)
            # neg_scores = torch.sum(ques_batch[-1] * neg_batch[-1], -1)
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
            ques_batch, _ = self.encoder(ques_batch, hidden)
            pos_batch, _ = self.encoder(path_batch, hidden)

            # Calculating dot score
            score = self.dense(torch.cat((ques_batch[-1], pos_batch[-1]), dim=1))

            '''
                Binary Cross Entropy loss function. @TODO: Check if we can give it 1/0 labels.
            '''
            loss = loss_fn(score, y_label)
            loss.backward()
            optimizer.step()
            return loss