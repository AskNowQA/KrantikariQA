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
import numpy as np
import time


device = torch.device("cuda")

def load_data(data,parameter_dict,pointwise,shuffle = False):
    # Loading training data
    td = dl.TrainingDataGenerator(data['train_questions'], data['train_pos_paths'], data['train_neg_paths'],
                                  parameter_dict['max_length'],
                                  parameter_dict['_neg_paths_per_epoch_train'], parameter_dict['batch_size']
                                  , parameter_dict['total_negative_samples'], pointwise=pointwise)
    return DataLoader(td, shuffle=shuffle)

def training_loop(training_model, parameter_dict,modeler,model,train_loader,
                  optimizer,loss_func, data, dataset, device, test_every, validate_every , pointwise = False, problem='core_chain'):

    model_save_location = aux.save_location(problem, training_model, dataset)

    train_loss = []
    valid_accuracy = []
    test_accuracy = []
    best_accuracy = 0

    # The Loop
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

            # Finally, train
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



        # Track training loss
        train_loss.append(sum(epoch_loss))

        # Run on validation set
        if epoch%validate_every == 0:
            valid_accuracy.append(aux.validation_accuracy(data['valid_questions'], data['valid_pos_paths'],
                                                      data['valid_neg_paths'],  modeler, model,device))
            if valid_accuracy[-1] >= best_accuracy:
                best_accuracy = valid_accuracy[-1]
                aux.save_model(model_save_location, model, model_name='encoder.torch'
                           , epochs=epoch, optimizer=optimizer, accuracy=best_accuracy)

        # Run on test set
        if epoch%test_every == 0:
            test_accuracy.append(aux.validation_accuracy(data['test_questions'], data['test_pos_paths'],
                                                     data['test_neg_paths'] , modeler, model,device))


        # Resample new negative paths per epoch and shuffle all data
        train_loader.dataset.shuffle()

        # Epoch level prints
        print("Time: %s\t" % (time.time() - epoch_time),
              "Loss: %s\t" % (sum(epoch_loss)),
              "Valdacc: %s\t" % (valid_accuracy[-1]),
               "Testacc: %s\n" % (test_accuracy[-1]))

    return train_loss, model, valid_accuracy, test_accuracy


#Model specific paramters
parameter_dict = {}
parameter_dict['max_length'] = 25
parameter_dict['hidden_size'] = 256
parameter_dict['number_of_layer'] = 2
parameter_dict['embedding_dim'] = 300
parameter_dict['vocab_size'] = 15000
parameter_dict['batch_size'] = 500
parameter_dict['bidirectional'] = True
parameter_dict['_neg_paths_per_epoch_train'] = 100
parameter_dict['_neg_paths_per_epoch_validation'] = 1000
parameter_dict['total_negative_samples'] = 1000
parameter_dict['epochs'] = 300
parameter_dict['dropout'] = 0.4


#Data loading specific parameters
COMMON_DATA_DIR = 'data/data/common'
_relations = aux.load_relation(COMMON_DATA_DIR)
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
               valid_pos_paths, valid_neg_paths, dummy_y_valid, test_questions, test_pos_paths, test_neg_paths,vectors = _a



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



pointwise = False
train_loader = load_data(data,parameter_dict,pointwise)
parameter_dict['vectors'] = data['vectors']
training_model = 'bilstm_dot'


if training_model == 'bilstm_dot':
    modeler = net.BiLstmDot( parameter_dict,device,_pointwise=pointwise, _debug=False)
    model = modeler.encoder
    optimizer = optim.Adam(list(model.parameters()))
    if not pointwise:
        loss_func = nn.MarginRankingLoss(margin=1,size_average=False)
        training_model = 'bilstm_dot'
    else:
        loss_func = nn.nn.BCELoss()
        training_model = 'bilstm_dot_pointwise'
    train_loss, model, valid_accuracy, test_accuracy = training_loop(training_model = training_model,
                                                                               parameter_dict = parameter_dict,
                                                                               modeler = modeler,
                                                                               model = model,
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


#rsync -avz --progress corechain.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
#rsync -avz --progress auxilary.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
# rsync -avz --progress network.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
# rsync -avz --progress components.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
# rsync -avz --progress data_loader.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
