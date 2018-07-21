from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader


import os,json,pickle,time
import numpy as np
import data_loader as dl

device = torch.device("cpu")


DEBUG = True


#helper Function
def load_relation():
    """
        Function used once to load the relations dictionary
        (which keeps the log of IDified relations, their uri and other things.)

    :param relation_file: str
    :return: dict
    """

    relations = pickle.load(open(os.path.join(COMMON_DATA_DIR, 'relations.pickle')))
    inverse_relations = {}
    for key in relations:
        value = relations[key]
        new_key = value[0]
        value[0] = key
        inverse_relations[new_key] = value

    return inverse_relations


# LSTM encoder.
class Encoder(nn.Module):

    def __init__(self, max_length, hidden_dim, number_of_layer, embedding_dim, vocab_size, bidirectional):
        super(Encoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = max_length, hidden_dim, embedding_dim, vocab_size
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.number_of_layer, bidirectional=self.bidirectional)

    def init_hidden(self, batch_size, device):
        # Return a new hidden layer variable for LSTM
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        if not self.bidirectional:
            return (torch.zeros((self.number_of_layer, batch_size, self.hidden_dim), device=device),
                    torch.zeros((self.number_of_layer, batch_size, self.hidden_dim), device=device))
        else:
            return (torch.zeros((2 * self.number_of_layer, batch_size, self.hidden_dim), device=device),
                    torch.zeros((2 * self.number_of_layer, batch_size, self.hidden_dim), device=device))

    def forward(self, x, h):
        # x is the input and h is the hidden state.
        batch_size = x.shape[0]

        if DEBUG: print("input/x shape is :", x.shape)
        if DEBUG: print("hidden state shape is :", h[0].shape)

        x_embedded = self.embedding_layer(x)
        if DEBUG: print("x_embedded transpose shape is :", x_embedded.transpose(1, 0).shape)

        #         output,h = self.lstm(x_embedded.view(-1,self.batch_size,self.embedding_dim),h)
        output, h = self.lstm(x_embedded.transpose(1, 0), h)
        if DEBUG: print("output shape is ", output.shape)
        if DEBUG: print("h[0] shape is ", h[0].shape, "h[1] shape is ", h[1].shape)

        return output, h


def train_procedure_bilstm_dot(ques_batch, pos_batch, neg_batch, dummy_y, model, optimizer, loss_fn, batch_size,
                     max_sequence_length):
    '''
        :params ques_batch: batch of question
        :params pos_batch: batch of corresponding positive paths
        :params neg_batch: batch of corresponding negative paths
        :params dummy_y:a batch of ones (same length as that of batch)
    '''

    hidden = model.init_hidden()
    ques_batch, _ = model(ques_batch.long(), hidden)
    pos_batch, _ = model(pos_batch.long(), hidden)
    neg_batch, _ = model(neg_batch.long(), hidden)

    pos_scores = torch.sum(ques_batch.view(batch_size, max_sequence_length, -1)[:, -1, :] *
                           pos_batch.view(batch_size, max_sequence_length, -1)[:, -1, :], -1)
    neg_scores = torch.sum(ques_batch.view(batch_size, max_sequence_length, -1)[:, -1, :] *
                           neg_batch.view(batch_size, max_sequence_length, -1)[:, -1, :], -1)

    '''
        If `y == 1` then it assumed the first input should be ranked higher
        (have a larger value) than the second input, and vice-versa for `y == -1`
    '''

    loss = loss_fn(pos_scores, neg_scores, dummy_y)
    loss.backward()
    optimizer.step()
    return loss


def training_loop(training_model, parameter_dict, train_questions, train_pos_paths, train_neg_paths):
    if training_model == 'bilstm_dot':
        # Model instantiation
        encoder = Encoder(parameter_dict['max_length'], parameter_dict['hidden_size']
                          , parameter_dict['number_of_layer'], parameter_dict['embedding_dim'],
                          parameter_dict['vocab_size'], parameter_dict['batch_size'],
                          bidirectional=parameter_dict['bidirectional'])
        # Loading training and validation data
        td = dl.TrainingDataGenerator(train_questions, train_pos_paths, train_neg_paths,
                                      parameter_dict['max_length'],
                                      parameter_dict['_neg_paths_per_epoch_train'], parameter_dict['batch_size']
                                      , parameter_dict['total_negative_samples'])
        vd = dl.ValidationDataset(train_questions, train_pos_paths, train_neg_paths,
                                      parameter_dict['max_length'],
                                      parameter_dict['_neg_paths_per_epoch_train'], parameter_dict['batch_size']
                                      , parameter_dict['total_negative_samples'])
        trainLoader = DataLoader(td)
        validationLoader = DataLoader(vd)

        # dummy_y needed for calcualting loss
        dummy_y = torch.ones(parameter_dict['batch_size'])
        optimizer = optim.Adam(list(encoder.parameters()))
        max_margin_loss = nn.MarginRankingLoss()
        train_loss_per_epoch = []
        validation_loss = []

        for epoch in range(parameter_dict['epochs']):
            print("Epoch: ", epoch, "/", parameter_dict['epochs'])
            epoch_loss = []
            epoch_time = time.time()

            for i_batch, sample_batched in enumerate(trainLoader):
                batch_time = time.time()
                batch_loss = train_procedure_bilstm_dot(ques_batch=np.reshape(sample_batched[0][0], (-1, parameter_dict['max_length'])),
                                        pos_batch=np.reshape(sample_batched[0][1], (-1, parameter_dict['max_length'])),
                                        neg_batch=np.reshape(sample_batched[0][2], (-1, parameter_dict['max_length'])),
                                        dummy_y=dummy_y,
                                        model=encoder,
                                        optimizer=optimizer,
                                        loss_fn=max_margin_loss,
                                        batch_size=parameter_dict['batch_size'],
                                        max_sequence_length=parameter_dict['max_length'])
                epoch_loss.append(batch_loss)
                print("Batch:\t%d" % i_batch, "/%d\t: " % (i_batch / parameter_dict['batch_size']),
                      "EpochLoss:\t%d" % (sum(epoch_loss)),
                      "BatchTimeElp\t%s" % (time.time() - batch_time),
                      "EpochTimeElp\t%s" % (time.time() - epoch_time),
                      end=None if i_batch + 1 == int(int(i_batch) / parameter_dict['batch_size']) else "\r")
            print("Time taken in epoch: %s" % (time.time() - epoch_time))

            train_loss_per_epoch.append(sum(epoch_loss))

        return train_loss_per_epoch


def train_bilstm_dot():
    #Model specific paramters
    parameter_dict = {}
    parameter_dict['max_length'] = 25
    parameter_dict['hidden_size'] = 15
    parameter_dict['number_of_layer'] = 10
    parameter_dict['embedding_dim'] = 30
    parameter_dict['vocab_size'] = 15000
    parameter_dict['batch_size'] = 50
    parameter_dict['bidirectional'] = True
    parameter_dict['_neg_paths_per_epoch_train'] = 10
    parameter_dict['_neg_paths_per_epoch_validation'] = 1000
    parameter_dict['total_negative_samples'] = 1000
    parameter_dict['epochs'] = 2


    #Data loading specific parameters
    COMMON_DATA_DIR = 'data/data/common'
    _relations = load_relation()
    _dataset = 'lcquad'
    _dataset_specific_data_dir = 'data/data/lcquad/'
    _model_specific_data_dir = 'data/data/core_chain_pairwise/lcquad/'
    _file = 'id_big_data.json'
    _max_sequence_length = parameter_dict['max_length']
    # _max_sequence_length = 15
    _neg_paths_per_epoch_train = parameter_dict['_neg_paths_per_epoch_train']
    _neg_paths_per_epoch_validation = parameter_dict['_neg_paths_per_epoch_validation']
    _training_split = .7
    _validation_split = .8
    _index = None



    _a = dl.load_data(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
                  _neg_paths_per_epoch_train,
                  _neg_paths_per_epoch_validation, _relations,
                  _index, _training_split, _validation_split, _model='core_chain_pairwise',_pairwise=True, _debug=True)
    train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, \
                   valid_pos_paths, valid_neg_paths, dummy_y_valid, test_questions, test_pos_paths, test_neg_paths = _a
    DEBUG = False
    training_loss = training_loop('bilstm_dot', parameter_dict,
                                  train_questions, train_pos_paths, train_neg_paths)
