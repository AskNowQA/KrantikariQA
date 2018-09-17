'''

    File loads macros for different models,datasets.

'''
import sys
if sys.version_info[0] == 3: import configparser as ConfigParser
else: import ConfigParser


def corechain_parameters(dataset, training_model, training_config, config_file ='configs/macros.cfg'):
    '''


    :param dataset: lcquad/qald
    :param trainiing_model: cnn_dot/bilst_dot etc.
    :param training_config: pairwise/pointwise
    :param config_file: the exact loaction from which the configuration file would be loaded
    :return:
    '''

    if not training_config:
        training_config = 'pairwise'
    else:
        training_config = 'pointwise'
    # Reading and setting up config parser
    config = ConfigParser.ConfigParser()
    config.readfp(open(config_file))

    parameter_dict = {}
    parameter_dict['dataset'] = dataset
    parameter_dict['max_length'] =  int(config.get('Commons','max_length'))
    parameter_dict['hidden_size'] = int(config.get(dataset, 'hidden_size'))
    parameter_dict['number_of_layer'] = int(config.get(dataset, 'number_of_layer'))
    parameter_dict['embedding_dim'] = int(config.get(dataset, 'embedding_dim'))
    parameter_dict['vocab_size'] = int(config.get(dataset, 'vocab_size'))
    parameter_dict['batch_size'] = int(config.get(dataset, 'batch_size'))
    parameter_dict['bidirectional'] = bool(config.get('Commons', 'bidirectional'))
    parameter_dict['_neg_paths_per_epoch_train'] = int(config.get(dataset, '_neg_paths_per_epoch_train'))
    parameter_dict['_neg_paths_per_epoch_validation'] = int(config.get(dataset, '_neg_paths_per_epoch_validation'))
    parameter_dict['total_negative_samples'] = int(config.get(dataset, 'total_negative_samples'))
    parameter_dict['epochs'] = int(config.get(training_config,'epochs'))
    parameter_dict['dropout'] = float(config.get(dataset, 'dropout'))
    parameter_dict['dropout_rec'] = float(config.get(dataset, 'dropout_rec'))
    parameter_dict['dropout_in'] = float(config.get(dataset, 'dropout_in'))
    parameter_dict['rel_pad'] = int(config.get(dataset, 'rel_pad'))
    parameter_dict['relsp_pad'] = int(config.get(dataset, 'relsp_pad'))
    parameter_dict['relrd_pad'] = int(config.get(dataset, 'relrd_pad'))

    if training_model == 'cnn_dot':
        parameter_dict['output_dim'] = int(config.get(dataset, 'output_dim'))
    parameter_dict['validate_every'] = int(config.get('Commons', 'validate_every'))
    parameter_dict['test_every'] = int(config.get('Commons', 'test_every'))

    return parameter_dict


def runtime_parameters(dataset, training_model, training_config, config_file ='configs/macros.cfg'):
    '''


    :param dataset: lcquad/qald
    :param trainiing_model: cnn_dot/bilst_dot etc.
    :param training_config: pairwise/pointwise
    :param config_file: the exact loaction from which the configuration file would be loaded
    :return:
    '''

    if not training_config:
        training_config = 'pairwise'
    else:
        training_config = 'pointwise'
    # Reading and setting up config parser
    config = ConfigParser.ConfigParser()
    config.readfp(open(config_file))

    parameter_dict = {}
    parameter_dict['dataset'] = dataset
    parameter_dict['max_length'] =  int(config.get('Commons','max_length'))
    parameter_dict['hidden_size'] = int(config.get(dataset, 'hidden_size'))
    parameter_dict['number_of_layer'] = int(config.get(dataset, 'number_of_layer'))
    parameter_dict['embedding_dim'] = int(config.get(dataset, 'embedding_dim'))
    parameter_dict['vocab_size'] = int(config.get(dataset, 'vocab_size'))
    parameter_dict['batch_size'] = int(config.get(dataset, 'batch_size'))
    parameter_dict['bidirectional'] = bool(config.get('Commons', 'bidirectional'))
    parameter_dict['_neg_paths_per_epoch_train'] = int(config.get(dataset, '_neg_paths_per_epoch_train'))
    parameter_dict['_neg_paths_per_epoch_validation'] = int(config.get(dataset, '_neg_paths_per_epoch_validation'))
    parameter_dict['total_negative_samples'] = int(config.get(dataset, 'total_negative_samples'))
    parameter_dict['epochs'] = int(config.get(training_config,'epochs'))
    parameter_dict['dropout'] = float(config.get(dataset, 'dropout'))
    parameter_dict['dropout_rec'] = float(config.get(dataset, 'dropout_rec'))
    parameter_dict['dropout_in'] = float(config.get(dataset, 'dropout_in'))
    parameter_dict['rel_pad'] = int(config.get(dataset, 'rel_pad'))
    parameter_dict['relrd_pad'] = int(config.get(dataset, 'relrd_pad'))
    parameter_dict['relsp_pad'] = int(config.get(dataset, 'relsp_pad'))
    if training_model == 'cnn_dot':
        parameter_dict['output_dim'] = int(config.get(dataset, 'output_dim'))

    parameter_dict['validate_every'] = int(config.get('Commons', 'validate_every'))
    parameter_dict['test_every'] = int(config.get('Commons', 'test_every'))
    parameter_dict['prune_corechain_candidates'] = bool(config.get('runtime', 'prune_corechain_candidates'))

    return parameter_dict