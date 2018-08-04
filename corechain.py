'''
    create loss function and training data and other necessary utilities

    TODO:
        > Add visaulization so as to understand the interplay of train vs Validation vs test accuracy (not really going to do it)
        > Add logs
            - Need to discuss it before implementing.
'''
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import  DataLoader

import data_loader as dl
import auxiliary as aux
import network as net
import argparse
import numpy as np
import time

from configs import config_loader as cl



#setting up device,model name and loss types.
device = torch.device("cuda")
training_model = 'bilstm_dot'
_dataset = 'lcquad'
pointwise = False
_train_over_validation = False


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
                y = torch.tensor(sample_batched[1],dtype = torch.float,device=device).view(-1)

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
        else:
            test_accuracy.append(0)
            best_test_accuracy = 0

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
                "Testacc: %s\n" % (test_accuracy[-1]),
              "BestValidAcc: %s\n" % (best_validation_accuracy),
              "BestTestAcc: %s\n" % (best_test_accuracy))

    return train_loss, modeler, valid_accuracy, test_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', action='store', dest='dataset',
                        help='dataset includes lcquad,qald')

    parser.add_argument('-model', action='store', dest='model',
                        help='name of the model to use')

    parser.add_argument('-pointwise', action='store', dest='pointwise',
                        help='to use pointwise training procedure make it true',default=False)

    parser.add_argument('-train_valid', action='store', dest='train_over_validation',
                        help='train over validation', default=False)

    parser.add_argument('-device', action='store', dest='device',
                        help='cuda for gpu else cpu', default='cuda')

    args = parser.parse_args()

    # setting up device,model name and loss types.
    device = torch.device(args.device)
    training_model = args.model
    _dataset = args.dataset
    pointwise = aux.to_bool(args.pointwise)
    _train_over_validation = aux.to_bool(args.train_over_validation)


    # #Model specific paramters
    if pointwise:
        training_config = 'pointwise'
    else:
        training_config = 'pairwise'

    parameter_dict = cl.corechain_parameters(dataset=_dataset,training_model=training_model,
                                             training_config=training_config,config_file='configs/macros.cfg')


    if _dataset == 'lcquad':
        test_every = parameter_dict['test_every']
    else:
        test_every = False
    validate_every = parameter_dict['validate_every']


    data = aux.load_data(_dataset=_dataset , _train_over_validation = _train_over_validation,
              _parameter_dict=parameter_dict, _relations =  _relations, _pointwise=pointwise, _device=device)


    train_loader = load_data(data,parameter_dict,pointwise)
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


    train_loss, modeler, valid_accuracy, test_accuracy = training_loop(training_model = training_model,
                                                                               parameter_dict = parameter_dict,
                                                                               modeler = modeler,
                                                                               train_loader = train_loader,
                                                                               optimizer=optimizer,
                                                                               loss_func=loss_func,
                                                                               data=data,
                                                                               dataset=parameter_dict['dataset'],
                                                                               device=device,
                                                                               test_every=test_every,
                                                                               validate_every=validate_every,
                                                                                pointwise=pointwise,
                                                                               problem='core_chain')

    print(valid_accuracy)
    print(test_accuracy)
    print("validation accuracy is , ", max(valid_accuracy))
    print("maximum test accuracy is , ", max(test_accuracy))
    print("correct test accuracy i.e test accuracy where validation is highest is ", test_accuracy[valid_accuracy.index(max(valid_accuracy))])
    #
    # rsync -avz --progress corechain.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
    # rsync -avz --progress auxiliary.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
    # rsync -avz --progress network.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
    # rsync -avz --progress components.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/
    # rsync -avz --progress data_loader.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/

    # rsync -avz --progress corechain.py priyansh@sda-srv04:/data/priyansh/new_kranti
    # rsync -avz --progress auxiliary.py priyansh@sda-srv04:/data/priyansh/new_kranti
    # rsync -avz --progress components.py priyansh@sda-srv04:/data/priansh/new_kranti
    # rsync -avz --progress network.py priyansh@sda-srv04:/data/priyansh/new_kranti