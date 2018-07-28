'''
    create loss function and training data and other necessary utilities
'''
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import  DataLoader

import data_loader as dl
import auxiliary as aux
import network as net
import ConfigParser
import numpy as np
import time



#Reading and setting up config parser
config = ConfigParser.ConfigParser()
config.readfp(open('configs/macros.cfg'))

#setting up device,model name and loss types.
device = torch.device("cuda")
training_model = 'bilstm_dot'
_dataset = 'lcquad'
pointwise = False


#Loading relations file.
COMMON_DATA_DIR = 'data/data/common'
_dataset_specific_data_dir = 'data/data/%(dataset)s/' % {'dataset': _dataset}
_relations = aux.load_relation(COMMON_DATA_DIR)
_word_to_id = aux.load_word_list(COMMON_DATA_DIR)


def load_data(data,parameter_dict,pointwise,shuffle = False):
    # Loading training data
    td = dl.TrainingDataGenerator(data['train_questions'], data['train_pos_paths'], data['train_neg_paths'],
                                  parameter_dict['max_length'],
                                  parameter_dict['_neg_paths_per_epoch_train'], parameter_dict['batch_size']
                                  , parameter_dict['total_negative_samples'], pointwise=pointwise)
    return DataLoader(td, shuffle=shuffle)

def training_loop(training_model, parameter_dict,modeler,train_loader,
                  optimizer,loss_func, data, dataset, device, test_every, validate_every , pointwise = False, problem='core_chain'):

    model_save_location = aux.save_location(problem, training_model, dataset)
    aux_save_information = {
        'epoch' : 0,
        'test_accuracy':0.0,
        'validation_accuracy':0.0,
        'parameter_dict':parameter_dict
    }
    train_loss = []
    valid_accuracy = []
    test_accuracy = []
    best_validation_accuracy = 0
    best_test_accuracy = 0

    ###############
    # Training Loop
    ###############
    for epoch in range(parameter_dict['epochs']):

        # Epoch start print
        print("Epoch: ", epoch, "/", parameter_dict['epochs'])

        # Bookkeeping variables
        epoch_loss = []
        epoch_time = time.time()

        # Loop for one batch
        for i_batch, sample_batched in enumerate(train_loader):

            # Bookkeeping and data preparation
            batch_time = time.time()

            if not pointwise:
                ques_batch = torch.tensor(np.reshape(sample_batched[0][0], (-1, parameter_dict['max_length'])),
                                          dtype=torch.long, device=device)
                pos_batch = torch.tensor(np.reshape(sample_batched[0][1], (-1, parameter_dict['max_length'])),
                                         dtype=torch.long, device=device)
                neg_batch = torch.tensor(np.reshape(sample_batched[0][2], (-1, parameter_dict['max_length'])),
                                         dtype=torch.long, device=device)

                data_batch = {
                    'ques_batch': ques_batch,
                    'pos_batch': pos_batch,
                    'neg_batch': neg_batch,
                    'y_label': data['dummy_y']
                }
            else:
                ques_batch = torch.tensor(np.reshape(sample_batched[0][0], (-1, parameter_dict['max_length'])),
                                          dtype=torch.long, device=device)
                path_batch = torch.tensor(np.reshape(sample_batched[0][1], (-1, parameter_dict['max_length'])),
                                         dtype=torch.long, device=device)
                y = torch._tensor(sample_batched[1],dtype = torch.long,device=device)

                data_batch = {
                    'ques_batch': ques_batch,
                    'path_batch': path_batch,
                    'y_label': y
                }

            # Train
            loss = modeler.train(data=data_batch,
                              optimizer=optimizer,
                              loss_fn=loss_func,
                              device=device)

            # Bookkeep the training loss
            epoch_loss.append(loss.item())

            print("Batch:\t%d" % i_batch, "/%d\t: " % (parameter_dict['batch_size']),
                  "%s" % (time.time() - batch_time),
                  "\t%s" % (time.time() - epoch_time),
                  "\t%s" % (str(loss.item())),
                  end=None if i_batch + 1 == int(int(i_batch) / parameter_dict['batch_size']) else "\n")

        # EPOCH LEVEL

        # Track training loss
        train_loss.append(sum(epoch_loss))

        if test_every:
            # Run on test set
            if epoch%test_every == 0:
                test_accuracy.append(aux.validation_accuracy(data['test_questions'], data['test_pos_paths'],
                                                         data['test_neg_paths'], modeler, device))
                if test_accuracy[-1] >= best_test_accuracy:
                    aux_save_information['test_accuracy'] = best_test_accuracy

        # Run on validation set
        if validate_every:
            if epoch%validate_every == 0:
                valid_accuracy.append(aux.validation_accuracy(data['valid_questions'], data['valid_pos_paths'],
                                                          data['valid_neg_paths'],  modeler, device))
                if valid_accuracy[-1] >= best_validation_accuracy:
                    best_validation_accuracy = valid_accuracy[-1]
                    aux_save_information['epoch'] = epoch
                    aux_save_information['validation_accuracy'] = best_validation_accuracy
                    aux.save_model(model_save_location, modeler, model_name='model.torch'
                               , epochs=epoch, optimizer=optimizer, accuracy=best_validation_accuracy, aux_save_information=aux_save_information)

        # Resample new negative paths per epoch and shuffle all data
        train_loader.dataset.shuffle()

        # Epoch level prints
        print("Time: %s\t" % (time.time() - epoch_time),
              "Loss: %s\t" % (sum(epoch_loss)),
              "Valdacc: %s\t" % (valid_accuracy[-1]),
               "Testacc: %s\n" % (test_accuracy[-1]))

    return train_loss, modeler, valid_accuracy, test_accuracy


# #Model specific paramters
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



TEMP = aux.data_loading_parameters(_dataset,parameter_dict)

_dataset_specific_data_dir,_model_specific_data_dir,_file,\
           _max_sequence_length,_neg_paths_per_epoch_train,_neg_paths_per_epoch_validation,_training_split,_validation_split,_index= TEMP

_a = dl.load_data(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
              _neg_paths_per_epoch_train,
              _neg_paths_per_epoch_validation, _relations,
              _index, _training_split, _validation_split, _model='core_chain_pairwise',_pairwise=not pointwise, _debug=True)


if _dataset == 'lcquad':
    train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, valid_pos_paths, valid_neg_paths, dummy_y_valid, test_questions, test_pos_paths, test_neg_paths,vectors = _a
else:
    print("warning: Test accuracy would not be calculated as the data has not been prepared.")
    train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, valid_pos_paths, valid_neg_paths, dummy_y_valid, vectors = _a
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


train_loader = load_data(data,parameter_dict,pointwise)
parameter_dict['vectors'] = data['vectors']


if training_model == 'bilstm_dot':
    modeler = net.BiLstmDot( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                         _device=device,_pointwise=pointwise, _debug=False)
    # optimizer = optim.Adam(list(model.parameters()))

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())))


    if not pointwise:
        loss_func = nn.MarginRankingLoss(margin=1,size_average=False)
        training_model = 'bilstm_dot'
    else:
        loss_func = nn.MSELoss()
        training_model = 'bilstm_dot_pointwise'
    train_loss, modeler, valid_accuracy, test_accuracy = training_loop(training_model = training_model,
                                                                               parameter_dict = parameter_dict,
                                                                               modeler = modeler,
                                                                               train_loader = train_loader,
                                                                               optimizer=optimizer,
                                                                               loss_func=loss_func,
                                                                               data=data,
                                                                               dataset='lcquad',
                                                                               device=device,
                                                                               test_every=5,
                                                                               validate_every=5,
                                                                                pointwise=pointwise,
                                                                               problem='core_chain')
    print(valid_accuracy)
    print(test_accuracy)
    print(max(valid_accuracy))
    print(max(test_accuracy))


if training_model == 'bilstm_dense':
    modeler = net.BiLstmDense( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                         _device=device,_pointwise=pointwise, _debug=False)

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters()))+
                           list(filter(lambda p: p.requires_grad, modeler.dense.parameters())))

    if not pointwise:
        loss_func = nn.MarginRankingLoss(margin=1,size_average=False)
        training_model = 'bilstm_dense'
    else:
        loss_func = nn.MSELoss()
        training_model = 'bilstm_dense_pointwise'
    train_loss, modeler, valid_accuracy, test_accuracy = training_loop(training_model = training_model,
                                                                               parameter_dict = parameter_dict,
                                                                               modeler = modeler,
                                                                               train_loader = train_loader,
                                                                               optimizer=optimizer,
                                                                               loss_func=loss_func,
                                                                               data=data,
                                                                               dataset='lcquad',
                                                                               device=device,
                                                                               test_every=5,
                                                                               validate_every=5,
                                                                                pointwise=pointwise,
                                                                               problem='core_chain')
    print(valid_accuracy)
    print(test_accuracy)
    print(max(valid_accuracy))
    print(max(test_accuracy))


if training_model == 'bilstm_densedot':
    modeler = net.BiLstmDenseDot( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                         _device=device,_pointwise=pointwise, _debug=False)

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())) +
                           list(filter(lambda p: p.requires_grad, modeler.dense.parameters())))

    if not pointwise:
        loss_func = nn.MarginRankingLoss(margin=1,size_average=False)
        training_model = 'bilstm_densedot'
    else:
        loss_func = nn.MSELoss()
        training_model = 'bilstm_densedot_pointwise'
    train_loss, modeler, valid_accuracy, test_accuracy = training_loop(training_model = training_model,
                                                                               parameter_dict = parameter_dict,
                                                                               modeler = modeler,
                                                                               train_loader = train_loader,
                                                                               optimizer=optimizer,
                                                                               loss_func=loss_func,
                                                                               data=data,
                                                                               dataset='lcquad',
                                                                               device=device,
                                                                               test_every=5,
                                                                               validate_every=5,
                                                                                pointwise=pointwise,
                                                                               problem='core_chain')
    print(valid_accuracy)
    print(test_accuracy)
    print(max(valid_accuracy))
    print(max(test_accuracy))



# rsync -avz --progress corechain.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
# rsync -avz --progress auxiliary.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
# rsync -avz --progress network.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
# rsync -avz --progress components.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
# rsync -avz --progress data_loader.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/

# rsync -avz --progress corechain.py priyansh@sda-srv04:/data/priyansh/new_kranti
# rsync -avz --progress auxiliary.py priyansh@sda-srv04:/data/priyansh/new_kranti
# rsync -avz --progress components.py priyansh@sda-srv04:/data/priyansh/new_kranti
# rsync -avz --progress network.py priyansh@sda-srv04:/data/priyansh/new_kranti