# Imports
from __future__ import print_function

# In-repo files
from utils import dbpedia_interface as db_interface
import data_loader as dl
import auxiliary as aux
import network as net
import corechain as cc

# Torch files
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import  DataLoader

# Other libs
import ConfigParser
import numpy as np
import time


device = torch.device("cuda")
np.random.seed(42)
torch.manual_seed(42)

# Important Macros

# Reading and setting up config parser
config = ConfigParser.ConfigParser()
config.readfp(open('configs/rdf_class.cfg'))

# Setting up device,model name and loss types.
training_model = 'bilstm_dot'
_dataset = 'lcquad'
pointwise = False

#Loading relations file.
COMMON_DATA_DIR = 'data/data/common'
_dataset_specific_data_dir = 'data/data/%(dataset)s/' % {'dataset': _dataset}
_relations = aux.load_relation(COMMON_DATA_DIR)
_word_to_id = aux.load_word_list(COMMON_DATA_DIR)

if __name__ == "__main__":

    # Fixing parameters
    parameter_dict = {}
    parameter_dict['max_length'] =  int(config.get(training_model,'max_length'))
    parameter_dict['hidden_size'] = int(config.get(training_model,'hidden_size'))
    parameter_dict['number_of_layer'] = int(config.get(training_model,'number_of_layer'))
    parameter_dict['embedding_dim'] = int(config.get(training_model,'embedding_dim'))
    parameter_dict['vocab_size'] = int(config.get(training_model,'vocab_size'))
    parameter_dict['batch_size'] = int(config.get(training_model,'batch_size'))
    parameter_dict['bidirectional'] = bool(config.get(training_model,'bidirectional'))
    parameter_dict['_neg_paths_per_epoch_train'] = int(config.get(training_model,'_neg_paths_per_epoch_train'))
    parameter_dict['_neg_paths_per_epoch_validation'] = int(config.get(training_model,'_neg_paths_per_epoch_validation'))
    parameter_dict['total_negative_samples'] = int(config.get(training_model,'total_negative_samples'))
    parameter_dict['epochs'] = int(config.get(training_model,'epochs'))
    parameter_dict['dropout'] = float(config.get(training_model,'dropout'))
    parameter_dict['dropout_rec'] = float(config.get(training_model,'dropout_rec'))
    parameter_dict['dropout_in'] = float(config.get(training_model,'dropout_in'))

    _dataset_specific_data_dir,\
        _model_specific_data_dir,\
        _file,\
        _max_sequence_length,\
        _neg_paths_per_epoch_train,\
        _neg_paths_per_epoch_validation,\
        _training_split,\
        _validation_split,\
        _index = aux.data_loading_parameters(_dataset,parameter_dict)

    parameter_dict['index'] = _index
    parameter_dict['training_split'] = _training_split
    parameter_dict['validation_split'] = _validation_split
    parameter_dict['dataset'] = _dataset

    # DBP instance is used just to get labels
    dbp = db_interface.DBPedia(_verbose=True, caching=False)

    _a = dl.load_data(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
                      _neg_paths_per_epoch_train,
                      _neg_paths_per_epoch_validation, _relations,
                      _index, _training_split, _validation_split,
                      _model='rdf_class_pairwise',_pairwise=not pointwise, _debug=True, _rdf=True)

    if _dataset == 'lcquad':
        train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, valid_pos_paths, \
            valid_neg_paths, dummy_y_valid, test_questions, test_pos_paths, test_neg_paths,vectors = _a
    else:
        print("warning: Test accuracy would not be calculated as the data has not been prepared.")
        train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, valid_pos_paths, \
            valid_neg_paths, dummy_y_valid, vectors = _a
        test_questions,test_neg_paths,test_pos_paths = None,None,None


    data = {}
    data['train_questions'] = train_questions
    data['train_pos_paths'] = train_pos_paths
    data['train_neg_paths'] = train_neg_paths
    data['valid_questions'] = valid_questions
    data['valid_pos_paths'] = valid_pos_paths
    data['valid_neg_paths'] = valid_neg_paths
    data['test_pos_paths'] = test_pos_paths
    data['test_neg_paths'] = test_neg_paths
    data['test_questions'] = test_questions
    data['vectors'] = vectors
    data['dummy_y'] = torch.ones(parameter_dict['batch_size'],device=device)


    train_loader = cc.load_data(data,parameter_dict,pointwise)
    parameter_dict['vectors'] = data['vectors']



    if training_model == 'bilstm_dot':
        modeler = net.BiLstmDot( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                             _device=device,_pointwise=pointwise, _debug=False)

        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())))

    if training_model == 'bilstm_dense':
        modeler = net.BiLstmDense( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                             _device=device,_pointwise=pointwise, _debug=False)

        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters()))+
                               list(filter(lambda p: p.requires_grad, modeler.dense.parameters())))

    if training_model == 'bilstm_densedot':
        modeler = net.BiLstmDenseDot( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                             _device=device,_pointwise=pointwise, _debug=False)

        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())) +
                               list(filter(lambda p: p.requires_grad, modeler.dense.parameters())))

    if training_model == 'cnn_dot':
        modeler = net.CNNDot( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                             _device=device,_pointwise=pointwise, _debug=False)

        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())))


    if not pointwise:
        loss_func = nn.MarginRankingLoss(margin=1,size_average=False)
    else:
        loss_func = nn.MSELoss()
        training_model += '_pointwise'


    train_loss, modeler, valid_accuracy, test_accuracy = cc.training_loop(training_model = training_model,
                                                                               parameter_dict = parameter_dict,
                                                                               modeler = modeler,
                                                                               train_loader = train_loader,
                                                                               optimizer=optimizer,
                                                                               loss_func=loss_func,
                                                                               data=data,
                                                                               dataset=parameter_dict['dataset'],
                                                                               device=device,
                                                                               test_every=5,
                                                                               validate_every=5,
                                                                                pointwise=pointwise,
                                                                               problem='rdf_class')

    print(valid_accuracy)
    print(test_accuracy)
    print(max(valid_accuracy))
    print(max(test_accuracy))
