'''
    create loss function and training data and other necessary utilities

    TODO:
        > Add visaulization so as to understand the interplay of train vs Validation vs test accuracy (not really going to do it)
        > Add logs
            - Need to discuss it before implementing.
'''
from __future__ import print_function

# Torch imports
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import  DataLoader

# Local imports
import data_loader as dl
import auxiliary as aux
import network as net

# Other libs
import numpy as np
import argparse
import time

from configs import config_loader as cl




#setting up device,model name and loss types.
device = torch.device("cpu")
training_model = 'bilstm_dot'
_dataset = 'lcquad'
pointwise = False
_train_over_validation = False


#Loading relations file.
COMMON_DATA_DIR = 'data/data/common'
_dataset_specific_data_dir = 'data/data/%(dataset)s/' % {'dataset': _dataset}
_inv_relations = aux.load_inverse_relation(COMMON_DATA_DIR)
_word_to_id = aux.load_word_list(COMMON_DATA_DIR)

# gloveid_to_embeddingid , embeddingid_to_gloveid, word_to_gloveid, gloveid_to_word = aux.load_embeddingid_gloveid()


def load_data(data,parameter_dict,pointwise,schema='default',shuffle = False):
    # Loading training data
    td = dl.TrainingDataGenerator(data,
                                  parameter_dict['max_length'],
                                  parameter_dict['_neg_paths_per_epoch_train'], parameter_dict['batch_size']
                                  , parameter_dict['total_negative_samples'], pointwise=pointwise,schema=schema)
    return DataLoader(td, shuffle=shuffle)


def curatail_padding(data,parameter_dict):

    '''
        Since schema is already implicitly defined/present in the parameter_dict['rel1_pad']
    '''

    data['test_neg_paths'] = data['test_neg_paths'][:, :, :parameter_dict['rel_pad']]
    data['test_pos_paths'] = data['test_pos_paths'][:, :parameter_dict['rel_pad']]

    if parameter_dict['schema'] == 'reldet':
        data['test_neg_paths_rel1_rd'] = data['test_neg_paths_rel1_rd'][:, :, :parameter_dict['rel1_pad']]
        data['test_neg_paths_rel2_rd'] = data['test_neg_paths_rel2_rd'][:, :, :parameter_dict['rel1_pad']]
        data['test_pos_paths_rel1_rd'] = data['test_pos_paths_rel1_rd'][:, :parameter_dict['rel1_pad']]
        data['test_pos_paths_rel2_rd'] = data['test_pos_paths_rel2_rd'][:, :parameter_dict['rel1_pad']]
    elif parameter_dict['schema'] == 'slotptr':
        data['test_neg_paths_rel1_sp'] = data['test_neg_paths_rel1_sp'][:, :, :parameter_dict['rel1_pad']]
        data['test_neg_paths_rel2_sp'] = data['test_neg_paths_rel2_sp'][:, :, :parameter_dict['rel1_pad']]
        data['test_pos_paths_rel1_sp'] = data['test_pos_paths_rel1_sp'][:, :parameter_dict['rel1_pad']]
        data['test_pos_paths_rel2_sp'] = data['test_pos_paths_rel2_sp'][:, :parameter_dict['rel1_pad']]

    data['valid_neg_paths'] = data['valid_neg_paths'][:, :, :parameter_dict['rel_pad']]
    data['valid_pos_paths'] = data['valid_pos_paths'][:, :parameter_dict['rel_pad']]

    if parameter_dict['schema'] == 'reldet':
        data['valid_neg_paths_rel1_rd'] = data['valid_neg_paths_rel1_rd'][:, :, :parameter_dict['rel1_pad']]
        data['valid_neg_paths_rel2_rd'] = data['valid_neg_paths_rel2_rd'][:, :, :parameter_dict['rel1_pad']]
        data['valid_pos_paths_rel1_rd'] = data['valid_pos_paths_rel1_rd'][:, :parameter_dict['rel1_pad']]
        data['valid_pos_paths_rel2_rd'] = data['valid_pos_paths_rel2_rd'][:, :parameter_dict['rel1_pad']]
    elif parameter_dict['schema'] == 'slotptr':
        data['valid_neg_paths_rel1_sp'] = data['valid_neg_paths_rel1_sp'][:, :, :parameter_dict['rel1_pad']]
        data['valid_neg_paths_rel2_sp'] = data['valid_neg_paths_rel2_sp'][:, :, :parameter_dict['rel1_pad']]
        data['valid_pos_paths_rel1_sp'] = data['valid_pos_paths_rel1_sp'][:, :parameter_dict['rel1_pad']]
        data['valid_pos_paths_rel2_sp'] = data['valid_pos_paths_rel2_sp'][:, :parameter_dict['rel1_pad']]

    return data


def training_loop(training_model, parameter_dict,modeler,train_loader,
                  optimizer,loss_func, data, dataset, device, test_every, validate_every , pointwise = False, problem='core_chain',curtail_padding_rel=True):

    model_save_location = aux.save_location(problem, training_model, dataset)
    aux_save_information = {
        'epoch' : 0,
        'test_accuracy':0.0,
        'validation_accura""cy':0.0,
        'parameter_dict':parameter_dict
    }
    train_loss = []
    valid_accuracy = []
    test_accuracy = []
    best_validation_accuracy = 0
    best_test_accuracy = 0

    if parameter_dict['schema'] == 'reldet':
        parameter_dict['rel1_pad'] =  parameter_dict['relrd_pad']
    elif parameter_dict['schema'] == 'slotptr':
        parameter_dict['rel1_pad'] = parameter_dict['relsp_pad']

        ###############
    # Training Loop
    ###############


    #Makes test data of appropriate shape
    print("the dataset is ", dataset)
    if curtail_padding_rel and dataset == 'lcquad':
        data = curatail_padding(data, parameter_dict)

    for epoch in range(parameter_dict['epochs']):

        # Epoch start print
        print("Epoch: ", epoch, "/", parameter_dict['epochs'])

        # Bookkeeping variables
        epoch_loss = []
        epoch_time = time.time()

        # Loop for one batch
        # tqdm_loop = tqdm(enumerate(train_loader))
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

                data['dummy_y'] = torch.ones(ques_batch.shape[0], device=device)
                if parameter_dict['schema'] != 'default':
                    pos_rel1_batch = torch.tensor(np.reshape(sample_batched[0][3], (-1, parameter_dict['max_length'])),
                                                  dtype=torch.long, device=device)
                    pos_rel2_batch = torch.tensor(np.reshape(sample_batched[0][4], (-1, parameter_dict['max_length'])),
                                                  dtype=torch.long, device=device)
                    neg_rel1_batch = torch.tensor(np.reshape(sample_batched[0][5], (-1, parameter_dict['max_length'])),
                                                  dtype=torch.long, device=device)
                    neg_rel2_batch = torch.tensor(np.reshape(sample_batched[0][6], (-1, parameter_dict['max_length'])),
                                                  dtype=torch.long, device=device)


                    data_batch = {
                        'ques_batch': ques_batch,
                        'pos_batch': pos_batch[:,:parameter_dict['rel_pad']],
                        'neg_batch': neg_batch[:,:parameter_dict['rel_pad']],
                        'y_label': data['dummy_y'],
                        'pos_rel1_batch': pos_rel1_batch[:,:parameter_dict['rel1_pad']],
                        'pos_rel2_batch':pos_rel2_batch[:,:parameter_dict['rel1_pad']],
                        'neg_rel1_batch':neg_rel1_batch[:,:parameter_dict['rel1_pad']],
                        'neg_rel2_batch' : neg_rel2_batch[:,:parameter_dict['rel1_pad']]
                    }

                else:

                    data_batch = {
                        'ques_batch': ques_batch,
                        'pos_batch': pos_batch[:,:parameter_dict['rel_pad']],
                        'neg_batch': neg_batch[:,:parameter_dict['rel_pad']],
                        'y_label': data['dummy_y']}

            else:
                ques_batch = torch.tensor(np.reshape(sample_batched[0][0], (-1, parameter_dict['max_length'])),
                                          dtype=torch.long, device=device)
                path_batch = torch.tensor(np.reshape(sample_batched[0][1], (-1, parameter_dict['max_length'])),
                                         dtype=torch.long, device=device)
                y = torch.tensor(sample_batched[1],dtype = torch.float,device=device).view(-1)


                if parameter_dict['schema'] != 'default':
                    path_rel1_batch = torch.tensor(np.reshape(sample_batched[0][2], (-1, parameter_dict['max_length'])),
                                                  dtype=torch.long, device=device)
                    path_rel2_batch = torch.tensor(np.reshape(sample_batched[0][3], (-1, parameter_dict['max_length'])),
                                                  dtype=torch.long, device=device)

                    data_batch = {
                        'ques_batch': ques_batch,
                        'path_batch': path_batch[:,:parameter_dict['rel_pad']],
                        'y_label': y,
                        'path_rel1_batch': path_rel1_batch[:,:parameter_dict['rel1_pad']],
                        'path_rel2_batch': path_rel2_batch[:,:parameter_dict['rel1_pad']]
                    }
                else:
                    data_batch = {
                        'ques_batch': ques_batch,
                        'path_batch': path_batch[:,:parameter_dict['rel_pad']],
                        'y_label': y
                    }



            loss = modeler.train(data=data_batch,
                              optimizer=optimizer,
                              loss_fn=loss_func,
                              device=device)

            # Bookkeep the training loss
            epoch_loss.append(loss.item())

            # tqdm_loop.desc("#"+str(i_batch)+"\tLoss:" + str(loss.item())[:min(5, len(str(loss.item())))])

            print("Batch:\t%d" % i_batch, "/%d\t: " % (parameter_dict['batch_size']),
                  "%s" % (time.time() - batch_time),
                  "\t%s" % (time.time() - epoch_time),
                  "\t%s" % (str(loss.item())),
                  end=None if i_batch + 1 == int(int(i_batch) / parameter_dict['batch_size']) else "\n")

        # EPOCH LEVEL

        # Track training loss
        train_loss.append(sum(epoch_loss))

        test_every = False
        if test_every:
            # Run on test set
            if epoch%test_every == 0:
                if parameter_dict['schema'] != 'default':
                    if parameter_dict['schema']  == 'slotptr':
                        test_accuracy.append(aux.validation_accuracy(data['test_questions'], data['test_pos_paths'],
                                                         data['test_neg_paths'],modeler, device,data['test_pos_paths_rel1_sp'],data['test_pos_paths_rel2_sp'],
                                                             data['test_neg_paths_rel1_sp'],data['test_neg_paths_rel2_sp']))
                    else:
                        test_accuracy.append(aux.validation_accuracy(data['test_questions'], data['test_pos_paths'],
                                                                     data['test_neg_paths'], modeler, device,
                                                                     data['test_pos_paths_rel1_rd'],
                                                                     data['test_pos_paths_rel2_rd'],
                                                                     data['test_neg_paths_rel1_rd'],
                                                                     data['test_neg_paths_rel2_rd']))
                else:
                    test_accuracy.append(aux.validation_accuracy(data['test_questions'], data['test_pos_paths'],
                                                                 data['test_neg_paths'], modeler, device))
                if test_accuracy[-1] >= best_test_accuracy:
                    best_test_accuracy = test_accuracy[-1]
                    aux_save_information['test_accuracy'] = best_test_accuracy
        else:
            test_accuracy.append(0)
            best_test_accuracy = 0

        # Run on validation set
        if validate_every:
            if epoch%validate_every == 0:
                if parameter_dict['schema'] != 'default':
                    if parameter_dict['schema'] == 'slotptr':
                        valid_accuracy.append(aux.validation_accuracy(data['valid_questions'], data['valid_pos_paths'],
                                                          data['valid_neg_paths'],  modeler, device, data['valid_pos_paths_rel1_sp'],data['valid_pos_paths_rel2_sp'],
                                                             data['valid_neg_paths_rel1_sp'],data['valid_neg_paths_rel2_sp']))
                    else:
                        valid_accuracy.append(aux.validation_accuracy(data['valid_questions'][:-1], data['valid_pos_paths'][:-1],
                                                                      data['valid_neg_paths'][:-1], modeler, device,
                                                                      data['valid_pos_paths_rel1_rd'][:-1],
                                                                      data['valid_pos_paths_rel2_rd'][:-1],
                                                                      data['valid_neg_paths_rel1_rd'][:-1],
                                                                      data['valid_neg_paths_rel2_rd'][:-1]))
                else:
                    valid_accuracy.append(aux.validation_accuracy(data['valid_questions'], data['valid_pos_paths'],
                                                                  data['valid_neg_paths'], modeler, device))
                if valid_accuracy[-1] > best_validation_accuracy:
                    print("MODEL WEIGHTS RIGHT NOW: ", modeler.get_parameter_sum())
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
                        help='dataset includes lcquad,qald',default = 'lcquad')

    parser.add_argument('-model', action='store', dest='model',
                        help='name of the model to use',default='bilstm_dot')

    parser.add_argument('-pointwise', action='store', dest='pointwise',
                        help='to use pointwise training procedure make it true',default=False)

    parser.add_argument('-train_valid', action='store', dest='train_over_validation',
                        help='train over validation', default=False)

    parser.add_argument('-device', action='store', dest='device',
                        help='cuda for gpu else cpu', default='cpu')

    parser.add_argument('-finetune', action='store', dest='finetune',
                        help='train over validation', default=False)

    parser.add_argument('-bidirectional', action='store', dest='bidirectional',
                        help='train over validation', default=True)

    args = parser.parse_args()

    # setting up device,model namenpp and loss types.
    device = torch.device(args.device)
    training_model = args.model
    _dataset = args.dataset
    pointwise = aux.to_bool(args.pointwise)
    _train_over_validation = aux.to_bool(args.train_over_validation)
    finetune = aux.to_bool(args.finetune)
    bidirectional = aux.to_bool(args.bidirectional)


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


    data = aux.load_data(_dataset=_dataset, _train_over_validation = _train_over_validation,
                         _parameter_dict=parameter_dict, _relations =  _inv_relations, _pointwise=pointwise, _device=device)

    if training_model == 'reldet':
        schema = 'reldet'
    elif training_model == 'slotptr' or training_model == 'slotptr_common_encoder' or training_model == 'slotptrortho':
        schema = 'slotptr'
    elif training_model == 'bilstm_dot_multiencoder':
        schema = 'default'
    else:
        schema = 'default'

    train_loader = load_data(data, parameter_dict, pointwise, schema=schema)
    # if training_model == 'bilstm_dot_ulmfit':
    #     _, data['vectors'] = vocab_master.load_ulmfit()
    parameter_dict['vectors'] = data['vectors']
    parameter_dict['schema'] = schema
    if not bidirectional:
        parameter_dict['bidirectional'] = False


    if training_model == 'bilstm_dot':
        if not finetune:
            print("********", parameter_dict['bidirectional'])
            modeler = net.BiLstmDot( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                                 _device=device,_pointwise=pointwise, _debug=False)

            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())))
        else:
            print("^^^^^^^^^^^^^^^^^^finetuning^^^^^^^^^^^^^^^^")
            if pointwise:
                model_path = 'data/models/core_chain/bilstm_dot_pointwise/lcquad/19/model.torch'
            else:
                model_path = 'data/models/core_chain/bilstm_dot/lcquad/30/model.torch'
            modeler = net.BiLstmDot(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                    _device=device, _pointwise=pointwise, _debug=False)

            modeler.load_from(model_path)
            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())),lr=0.0001)

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

    if training_model == 'decomposable_attention':
        if not finetune:
            modeler = net.DecomposableAttention(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                                _device=device, _pointwise=pointwise, _debug=False)

            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())) +
                                   list(filter(lambda p: p.requires_grad, modeler.scorer.parameters())))
        else:
            if not pointwise:
                model_path = 'data/models/core_chain/decomposable_attention/lcquad/71/model.torch'
            else:
                model_path = 'data/models/core_chain/decomposable_attention_pointwise/lcquad/8/model.torch'

            modeler = net.DecomposableAttention(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                                _device=device, _pointwise=pointwise, _debug=False)

            modeler.load_from(model_path)
            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())) +
                                   list(filter(lambda p: p.requires_grad, modeler.scorer.parameters())),lr=0.0001)

    if training_model == 'reldet':
        modeler = net.RelDetection(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                            _device=device, _pointwise=pointwise, _debug=False)
        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())))
    # if False:
    # # if training_model == 'slotptr':00
    #     modeler = net.SlotPointerModel(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
    #                                         _device=device, _pointwise=pointwise, _debug=False)
    #     optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters())) +
    #                            list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())) +
    #                            list(filter(lambda p: p.requires_grad, modeler.comparer.parameters())))
    if training_model == 'slotptr':

        if not finetune:
            modeler = net.QelosSlotPointerModel(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                                _device=device, _pointwise=pointwise, _debug=False)
            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters())) +
                                   list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())), weight_decay=0.0001)
        else:
            print("^^^^^^^^^^^^finetuning^^^^^^^")
            if pointwise:
                model_path = 'data/models/core_chain/slotptr_pointwise/lcquad/8/model.torch'
            else:
                model_path = 'data/models/core_chain/slotptr/lcquad/71/model.torch'
            modeler = net.QelosSlotPointerModel(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                                               _device=device, _pointwise=pointwise, _debug=False)
            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters())) +
                                   list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())),
                                   weight_decay=0.0001,lr=0.0001)
            modeler.load_from(model_path)


    if training_model == 'slotptrortho':

        if not finetune:
            modeler = net.QelosSlotPointerModelOrthogonal(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                                _device=device, _pointwise=pointwise, _debug=False)
            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters())) +
                                   list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())), weight_decay=0.0001)
        else:
            modeler = net.QelosSlotPointerModelOrthogonal(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                                               _device=device, _pointwise=pointwise, _debug=False)
            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters())) +
                                   list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())),
                                   weight_decay=0.0001,lr=0.0001)
            # modeler.load_from(model_path)


    if training_model == 'bilstm_dot_skip':
        modeler = net.BiLstmDot_skip( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                             _device=device,_pointwise=pointwise, _debug=False)

        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())))

    if training_model == 'bilstm_dot_multiencoder':
        print(schema)
        modeler = net.BiLstmDot_multiencoder( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                             _device=device,_pointwise=pointwise, _debug=False)

        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())) +
                               list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters())))

    if training_model == 'bilstm_dot_ulmfit':
        modeler = net.BiLstmDot_ulmfit( _parameter_dict = parameter_dict,_word_to_id=_word_to_id,
                                             _device=device,_pointwise=pointwise, _debug=False)

        modeler.freeze_layer(modeler.encoder)
        modeler.unfreeze_layer(modeler.encoder.rnns[-1])
        # +
        # list(filter(lambda p: p.requires_grad, modeler.linear.parameters()))
        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())),lr=0.001)

    if training_model == 'slotptr_common_encoder':
        print("*************",parameter_dict['bidirectional'])


        if not finetune:
            modeler = net.QelosSlotPointerModel_common_encoder(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                                _device=device, _pointwise=pointwise, _debug=False)
            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters())) +
                                   list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())), weight_decay=0.0001)
        else:
            print("^^^^^^^^^^^^finetuning^^^^^^^")
            if pointwise:
                model_path = 'data/models/core_chain/slotptr_common_encoder_pointwise/lcquad/1/model.torch'
            else:
                model_path = 'data/models/core_chain/slotptr_common_encoder/lcquad/5/model.torch'
            modeler = net.QelosSlotPointerModel_common_encoder(_parameter_dict=parameter_dict, _word_to_id=_word_to_id,
                                                               _device=device, _pointwise=pointwise, _debug=False)
            optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters())) +
                                   list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())),
                                   weight_decay=0.0001)
            modeler.load_from(model_path)


    if not pointwise:
        loss_func = nn.MarginRankingLoss(margin=1,size_average=False)
    else:
        loss_func = nn.BCEWithLogitsLoss()
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