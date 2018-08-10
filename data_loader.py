'''
    Loads data from the folder and creates a test/train/validation splits.
'''

import os
import json
import math
import numpy as np
import ConfigParser
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences



# Custom imports
from utils import embeddings_interface
from utils import prepare_vocab_continous as vocab_master
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils

import auxiliary as aux


config = ConfigParser.RawConfigParser()
config.read('configs/macros.cfg')

SEED = config.getint('Commons', 'seed')
dbp = db_interface.DBPedia(_verbose=True, caching=False)
COMMON_DATA_DIR = 'data/data/common'

def load_data(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
              _neg_paths_per_epoch_train,
              _neg_paths_per_epoch_validation, _relations,
              _index, _training_split, _validation_split, _model='core_chain_pairwise',_pairwise=True, _debug=True,_rdf=False,
              _schema = 'default'):
    '''


    :param _dataset:lcquad
    :param _dataset_specific_data_dir: './data/data/%(dataset)s/'
    :param _model_specific_data_dir: 'data/data/core_chain_pairwise/lcquad/'
    :param _max_sequence_length:
    :param _model: Use default
    :param _relations: object generated after passing it through network.load_relation()
    :param _index:_data
    :param _training_split: .70
    :param _validation_split: .80
    :param _pairwise: not needed anymore.
    :param _file: id_big_data.json
    :return:



        if _index is passed it breaks the data into two sets --> [input,validation]

        schema decides the kind of data required
        > default - used by all the model apart from slotptr network and reldet
            returns vectors, questions, pos_paths, neg_paths
        >slotptr - used by slot pointer mechanims
            returns vectors, questions, pos_paths, neg_paths


    '''
    _pairwise = True
    if _pairwise:
        if not _rdf:
            vectors, questions, pos_paths, neg_paths, pos_paths_rel1_sp, pos_paths_rel2_sp, neg_paths_rel1_sp, neg_paths_rel2_sp, \
            pos_paths_rel1_rd, pos_paths_rel2_rd, neg_paths_rel1_rd, neg_paths_rel2_rd = create_dataset_pairwise(file=_file, max_sequence_length=_max_sequence_length, relations=_relations, _dataset=_dataset, _dataset_specific_data_dir=_dataset_specific_data_dir,
            _model_specific_data_dir = _model_specific_data_dir, _model = 'core_chain_pairwise')


        else:
            vectors, questions, pos_paths, neg_paths = create_dataset_rdf(file=_file, max_sequence_length=_max_sequence_length, _dataset=_dataset, _dataset_specific_data_dir=_dataset_specific_data_dir,
            _model_specific_data_dir=_model_specific_data_dir)

    '''
        Making sure that positive path is not the part of negative paths.
    '''
    data = {}
    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                    neg_paths_rel1_sp[i][j] = neg_paths_rel1_sp[i][j + 10]
                    neg_paths_rel2_sp[i][j] = neg_paths_rel2_sp[i][j + 10]
                    neg_paths_rel1_rd[i][j] = neg_paths_rel1_rd[i][j + 10]
                    neg_paths_rel2_rd[i][j] = neg_paths_rel2_rd[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
                    neg_paths_rel1_sp[i][j] = neg_paths_rel1_sp[i][0]
                    neg_paths_rel2_sp[i][j] = neg_paths_rel2_sp[i][0]
                    neg_paths_rel1_rd[i][j] = neg_paths_rel1_rd[i][0]
                    neg_paths_rel2_rd[i][j] = neg_paths_rel2_rd[i][0]
    if counter > 0:
        print(counter)
        print("Critical condition needs to be entered")

    np.random.seed(SEED)  # Random train/test splits stay the same between runs

    if _index:
        '''
            split at the points with explicit index
        '''

    # Divide the data into diff blocks
    if _index:
        split_point = lambda x: _index + 1
    else:
        split_point = lambda x: int(len(x) * _training_split)

    def training_split(x):
        return x[:split_point(x)]

    def validation_split(x):
        if _index:
            return x[split_point(x):]
        return x[split_point(x):int(_validation_split * len(x))]

    def testing_split(x):
        return x[int(_validation_split * len(x)):]

    data['train_pos_paths'] = training_split(pos_paths)
    data['train_pos_paths_rel1_sp'] = training_split(pos_paths_rel1_sp)
    data['train_pos_paths_rel2_sp'] = training_split(pos_paths_rel2_sp)
    data['train_pos_paths_rel1_rd'] = training_split(pos_paths_rel1_rd)
    data['train_pos_paths_rel2_rd'] = training_split(pos_paths_rel2_rd)

    data['train_neg_paths'] = training_split(neg_paths)
    data['train_neg_paths_rel1_sp'] = training_split(neg_paths_rel1_sp)
    data['train_neg_paths_rel2_sp']= training_split(neg_paths_rel2_sp)
    data['train_neg_paths_rel1_rd'] = training_split(neg_paths_rel1_rd)
    data['train_neg_paths_rel2_rd'] = training_split(neg_paths_rel2_rd)

    data['train_questions'] = training_split(questions)

    data['valid_pos_paths'] = validation_split(pos_paths)
    data['valid_pos_paths_rel1_sp'] = validation_split(pos_paths_rel1_sp)
    data['valid_pos_paths_rel2_sp'] = validation_split(pos_paths_rel2_sp)
    data['valid_pos_paths_rel1_rd'] = validation_split(pos_paths_rel1_rd)
    data['valid_pos_paths_rel2_rd'] = validation_split(pos_paths_rel2_rd)

    data['valid_neg_paths'] = validation_split(neg_paths)
    data['valid_neg_paths_rel1_sp'] = validation_split(neg_paths_rel1_sp)
    data['valid_neg_paths_rel2_sp'] = validation_split(neg_paths_rel2_sp)
    data['valid_neg_paths_rel1_rd'] = validation_split(neg_paths_rel1_rd)
    data['valid_neg_paths_rel2_rd'] = validation_split(neg_paths_rel2_rd)


    data['valid_questions'] = validation_split(questions)

    if not _index:
        data['test_pos_paths'] = testing_split(pos_paths)
        data['test_pos_paths_rel1_sp'] = testing_split(pos_paths_rel1_sp)
        data['test_pos_paths_rel2_sp'] = testing_split(pos_paths_rel2_sp)
        data['test_pos_paths_rel1_rd'] = testing_split(pos_paths_rel1_rd)
        data['test_pos_paths_rel2_rd'] = testing_split(pos_paths_rel2_rd)


        data['test_neg_paths'] = testing_split(neg_paths)
        data['test_neg_paths_rel1_sp'] = testing_split(neg_paths_rel1_sp)
        data['test_neg_paths_rel2_sp'] = testing_split(neg_paths_rel2_sp)
        data['test_neg_paths_rel1_rd'] = testing_split(neg_paths_rel1_rd)
        data['test_neg_paths_rel2_rd'] = testing_split(neg_paths_rel2_rd)

        data['test_questions'] = testing_split(questions)


    data['dummy_y_train'] = np.zeros(len(data['train_questions']) * _neg_paths_per_epoch_train)
    data['dummy_y_valid'] = np.zeros(len(data['valid_questions']) * (_neg_paths_per_epoch_validation + 1))
    data['vectors'] = vectors

    if _debug:
        print(data['train_questions'].shape)
        print(data['train_pos_paths'].shape)
        print(data['train_neg_paths'].shape)

        print(data['valid_questions'].shape)
        print(data['valid_pos_paths'].shape)
        print(data['valid_neg_paths'].shape)

        if not _index:
            print(data['test_questions'].shape)
            print(data['test_pos_paths'].shape)
            print(data['test_neg_paths'].shape)




    return data

    # if _index:
    #     print "at _index locations"
    #     if _schema == 'default':
    #         return train_questions, train_pos_paths, train_neg_paths, dummy_y_train, \
    #            valid_questions, valid_pos_paths, valid_neg_paths, dummy_y_valid, vectors
    #     elif _schema == 'slotptr':
    #         return train_questions, train_pos_paths, train_neg_paths, dummy_y_train, \
    #            valid_questions, valid_pos_paths, valid_neg_paths, dummy_y_valid, vectors, \
    #             train_pos_paths_rel1_sp, train_pos_paths_rel2_sp, train_neg_paths_rel1_sp, train_neg_paths_rel2_sp, \
    #                valid_pos_paths_rel1_sp, valid_pos_paths_rel2_sp, valid_neg_paths_rel1_sp, valid_neg_paths_rel2_sp
    #     elif _schema == 'reldet':
    #         return train_questions, train_pos_paths, train_neg_paths, dummy_y_train, \
    #            valid_questions, valid_pos_paths, valid_neg_paths, dummy_y_valid, vectors, \
    #             train_pos_paths_rel1_rd, train_pos_paths_rel2_rd, train_neg_paths_rel1_rd, train_neg_paths_rel2_rd, \
    #                valid_pos_paths_rel1_rd, valid_pos_paths_rel2_rd, valid_neg_paths_rel1_rd, valid_neg_paths_rel2_rd
    #
    #
    # else:
    #     if _schema == 'slotptr':
    #         return train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, \
    #            valid_pos_paths, valid_neg_paths, dummy_y_valid, test_questions, test_pos_paths, test_neg_paths, vectors, \
    #                train_pos_paths_rel1_sp, train_pos_paths_rel2_sp, train_neg_paths_rel1_sp, train_neg_paths_rel2_sp, \
    #                    valid_pos_paths_rel1_sp, valid_pos_paths_rel2_sp, valid_neg_paths_rel1_sp, valid_neg_paths_rel2_sp, \
    #                        test_pos_paths_rel1_sp, test_pos_paths_rel2_sp, test_neg_paths_rel1_sp, test_neg_paths_rel2_sp
    #
    #
    #     elif _schema == 'default':
    #         return train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, \
    #            valid_pos_paths, valid_neg_paths, dummy_y_valid, test_questions, test_pos_paths, test_neg_paths, vectors
    #
    #     elif _schema == 'reldet':
    #         return train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, \
    #                valid_pos_paths, valid_neg_paths, dummy_y_valid, test_questions, test_pos_paths, test_neg_paths, vectors, \
    #                     train_pos_paths_rel1_rd, train_pos_paths_rel2_rd, train_neg_paths_rel1_rd, train_neg_paths_rel2_rd, \
    #                         valid_pos_paths_rel1_rd, valid_pos_paths_rel2_rd, valid_neg_paths_rel1_rd, valid_neg_paths_rel2_rd, \
    #                             test_pos_paths_rel1_rd, test_pos_paths_rel2_rd, test_neg_paths_rel1_rd, test_neg_paths_rel2_rd





def relation_table_lookup(lookup_str,glove_id_sf_to_glove_id_rel):
    '''
        given an np array of durface form of relation it gives a glove id for the whole relation string(list)
    '''
    if list(lookup_str)[0] == 0:
        return None

    key = str(list(lookup_str)[1:]) # 1 ownwards because first is the sign
    return glove_id_sf_to_glove_id_rel[key]

def create_relation_lookup_table(COMMON_DATA_DIR):
    '''

        Creates a lookup table with key being
            glove id of the surface form of relation (str(list(numpy))) and the whole relation id
    :param COMMON_DATA_DIR:
    :return:
    '''
    relation = aux.load_relation(COMMON_DATA_DIR)
    glove_id_sf_to_glove_id_rel = {}
    for keys in relation:
        k = str(list(relation[keys][3]))
        glove_id_sf_to_glove_id_rel[k] = list(relation[keys][4])

    return glove_id_sf_to_glove_id_rel


def get_glove_embeddings():
    from utils.embeddings_interface import __check_prepared__
    __check_prepared__('glove')

    from utils.embeddings_interface import glove_embeddings
    return glove_embeddings


def remove_positive_path(positive_path, negative_paths):
    '''
        Removes positive path from a set of negative paths.
    '''
    counter = 0
    new_negative_paths = []
    for i in range(0, len(negative_paths)):
        if not np.array_equal(negative_paths[i], positive_path):
            new_negative_paths.append(np.asarray(negative_paths[i]))
        else:
            counter += 1
            # print counter
    return new_negative_paths


def create_dataset_pairwise(file, max_sequence_length, relations, _dataset, _dataset_specific_data_dir,
                            _model_specific_data_dir
                            , _model='core_chain_pairwise',schema='default'):
    """
        This file is meant to create data for core-chain ranking ONLY.

    :param file: id_big_data file
    :param max_sequence_length: for padding/cropping
    :param relations: the relations file to backtrack and look up shit.
    :return:

    schema decides the kind of data required
        > default - used by all the model apart from slotptr network and reldet
            returns vectors, questions, pos_paths, neg_paths
        >slotptr - used by slot pointer mechanims
            returns vectors, questions, pos_paths, neg_paths
    """

    try:
        with open(os.path.join(_model_specific_data_dir % {'dataset': _dataset, 'model': _model},
                               file + ".mapped.npz")) as data:
            dataset = np.load(data)
            questions, pos_paths, neg_paths, \
            pos_paths_rel1_sp, pos_paths_rel2_sp,neg_paths_rel1_sp, neg_paths_rel2_sp, \
            pos_paths_rel1_rd, pos_paths_rel2_rd, neg_paths_rel1_rd, neg_paths_rel2_rd = dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], \
                                                                                         dataset['arr_3'], dataset['arr_4'], dataset['arr_5'], dataset['arr_6'], \
                                                                                         dataset['arr_7'], dataset['arr_8'],dataset['arr_9'], dataset['arr_10']



            vocab, vectors = vocab_master.load()
            #TODO: return everything.
            return vectors,questions, pos_paths, neg_paths, \
            pos_paths_rel1_sp, pos_paths_rel2_sp,neg_paths_rel1_sp, neg_paths_rel2_sp, \
            pos_paths_rel1_rd, pos_paths_rel2_rd, neg_paths_rel1_rd, neg_paths_rel2_rd

    except (EOFError, IOError) as e:
        with open(os.path.join(_dataset_specific_data_dir % {'dataset': _dataset}, file)) as fp:
            dataset = json.load(fp)
            glove_id_sf_to_glove_id_rel = create_relation_lookup_table(COMMON_DATA_DIR)

            ignored = []
            dummy_path = [0]

            pos_paths_rel1_sp = []
            pos_paths_rel2_sp = []
            neg_paths_rel1_sp = []
            neg_paths_rel2_sp = []

            pos_paths_rel1_rd = []
            pos_paths_rel2_rd = []
            neg_paths_rel1_rd = []
            neg_paths_rel2_rd = []


            pos_paths = []
            for i in dataset:
                path_id = i['parsed-data']['path_id']
                positive_path = []
                try:
                    for p in path_id:
                        positive_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p[0])]
                        positive_path += relations[int(p[1:])][3].tolist()
                except (TypeError, ValueError) as e:
                    ignored.append(i)
                    continue
                pos_paths.append(positive_path)


            '''     
                        $$$$$$$$$$$$$$$$$$$$$$$$$$$
                        $$$$$$$$$$$$$$$$$$$$$$$$$$$
            
                 The question id is not correct. It sis fucking out of sync 
            
                        $$$$$$$$$$$$$$$$$$$$$$$$$$$$
                        $$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                        positive_path += relations[int(p[1:])][3].tolist()

                
                To solve this issue in qald file. Execute scripts/update_questionid_qald file from ipython
                This remaps the question['uri'] to glove(question['uri']). This is a hack, but something clear needs to be worked out.
                
                
            '''

            questions = [i['uri']['question-id'] for i in dataset if i not in ignored]
            questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')

            neg_paths = []
            for i in range(0, len(pos_paths)):
                if i in ignored:
                    continue
                datum = dataset[i]
                negative_paths_id = datum['uri']['hop-2-properties'] + datum['uri']['hop-1-properties']
                np.random.shuffle(negative_paths_id)
                negative_paths = []
                for neg_path in negative_paths_id:
                    negative_path = []
                    for p in neg_path:
                        try:
                            negative_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p)]
                        except ValueError:
                            negative_path += relations[int(p)][3].tolist()
                    negative_paths.append(negative_path)
                negative_paths = remove_positive_path(pos_paths[i], negative_paths)
                try:
                    negative_paths = np.random.choice(negative_paths, 1000)
                except ValueError:
                    if len(negative_paths) == 0:
                        try:
                            negative_paths = neg_paths[-1]
                            print("Using previous question's paths for this since no neg paths for this question.")
                        except IndexError:
                            print("at index error. Moving forward due to a hack")
                            negative_paths = np.asarray([1])
                    else:
                        index = np.random.randint(0, len(negative_paths), 1000)
                        negative_paths = np.array(negative_paths)
                        negative_paths = negative_paths[index]
                neg_paths.append(negative_paths)


            special_char = [embeddings_interface.vocabularize(['+']), embeddings_interface.vocabularize(['-'])]
            for pps in pos_paths:
                p1, p2 = break_path(pps, special_char)
                pos_paths_rel1_sp.append(p1)
                pos_paths_rel1_rd.append(relation_table_lookup(p1,glove_id_sf_to_glove_id_rel))
                if p2 is not None:
                    pos_paths_rel2_sp.append(p2)
                    pos_paths_rel2_rd.append(relation_table_lookup(p2,glove_id_sf_to_glove_id_rel))
                else:
                    pos_paths_rel2_sp.append(dummy_path)
                    pos_paths_rel2_rd.append(dummy_path)

            for npps in neg_paths:
                temp_neg_paths_rel1_sp = []
                temp_neg_paths_rel2_sp = []
                temp_neg_paths_rel1_rd = []
                temp_neg_paths_rel2_rd = []

                for npp in npps:
                    p1, p2 = break_path(npp, special_char)
                    temp_neg_paths_rel1_sp.append(p1)
                    temp_neg_paths_rel1_rd.append(relation_table_lookup(p1,glove_id_sf_to_glove_id_rel))
                    if p2 is not None:
                        temp_neg_paths_rel2_sp.append(p2)
                        temp_neg_paths_rel2_rd.append(relation_table_lookup(p2, glove_id_sf_to_glove_id_rel))
                    else:
                        temp_neg_paths_rel2_sp.append(dummy_path)
                        temp_neg_paths_rel2_rd.append(dummy_path)
                neg_paths_rel1_sp.append(temp_neg_paths_rel1_sp)
                neg_paths_rel2_sp.append(temp_neg_paths_rel2_sp)
                neg_paths_rel1_rd.append(temp_neg_paths_rel1_rd)
                neg_paths_rel2_rd.append(temp_neg_paths_rel2_rd)


            for i in range(0, len(neg_paths)):
                neg_paths[i] = pad_sequences(neg_paths[i], maxlen=max_sequence_length, padding='post')
                neg_paths_rel1_sp[i] = pad_sequences(neg_paths_rel1_sp[i], maxlen=max_sequence_length, padding='post')
                neg_paths_rel2_sp[i] = pad_sequences(neg_paths_rel2_sp[i], maxlen=max_sequence_length, padding='post')
                neg_paths_rel1_rd[i] = pad_sequences(neg_paths_rel1_rd[i], maxlen=max_sequence_length, padding='post')
                neg_paths_rel2_rd[i] = pad_sequences(neg_paths_rel2_rd[i], maxlen=max_sequence_length, padding='post')
            neg_paths = np.asarray(neg_paths)
            neg_paths_rel1_sp = np.asarray(neg_paths_rel1_sp)
            neg_paths_rel2_sp = np.asarray(neg_paths_rel2_sp)
            neg_paths_rel1_rd = np.asarray(neg_paths_rel1_rd)
            neg_paths_rel2_rd = np.asarray(neg_paths_rel2_rd)

            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
            pos_paths_rel1_sp = pad_sequences(pos_paths_rel1_sp, maxlen=max_sequence_length, padding='post')
            pos_paths_rel2_sp = pad_sequences(pos_paths_rel2_sp, maxlen=max_sequence_length, padding='post')
            pos_paths_rel1_rd = pad_sequences(pos_paths_rel1_rd, maxlen=max_sequence_length, padding='post')
            pos_paths_rel2_rd = pad_sequences(pos_paths_rel2_rd, maxlen=max_sequence_length, padding='post')


            vocab, vectors = vocab_master.load()

            # Map everything
            unks_counter = 0
            # number of unks
            for i in range(len(questions)):
                for index in range(len(questions[i])):
                    try:
                        questions[i][index] = vocab[questions[i][index]]
                    except KeyError:
                        unks_counter = unks_counter + 1
                        questions[i][index] = 1
                # questions[i] = np.asarray([vocab[key] for key in questions[i]])
                pos_paths[i] = np.asarray([vocab[key] for key in pos_paths[i]])
                pos_paths_rel1_sp[i] = np.asarray([vocab[key] for key in pos_paths_rel1_sp[i]])
                pos_paths_rel2_sp[i] = np.asarray([vocab[key] for key in pos_paths_rel2_sp[i]])
                pos_paths_rel1_rd[i] = np.asarray([vocab[key] for key in pos_paths_rel1_rd[i]])
                pos_paths_rel2_rd[i] = np.asarray([vocab[key] for key in pos_paths_rel2_rd[i]])

                for j in range(len(neg_paths[i])):
                    neg_paths[i][j] = np.asarray([vocab[key] for key in neg_paths[i][j]])
                    neg_paths_rel1_sp[i][j] = np.asarray([vocab[key] for key in neg_paths_rel1_sp[i][j]])
                    neg_paths_rel2_sp[i][j] = np.asarray([vocab[key] for key in neg_paths_rel2_sp[i][j]])
                    neg_paths_rel1_rd[i][j] = np.asarray([vocab[key] for key in neg_paths_rel1_rd[i][j]])
                    neg_paths_rel2_rd[i][j] = np.asarray([vocab[key] for key in neg_paths_rel2_rd[i][j]])

            print("places where glove id exists and not in vecotrs ", unks_counter)
            with open(os.path.join(_model_specific_data_dir % {'dataset': _dataset, 'model': _model},
                                   file + ".mapped.npz"), "w+") as data:
                np.savez(data, questions, pos_paths, neg_paths,
                         pos_paths_rel1_sp, pos_paths_rel2_sp, neg_paths_rel1_sp, neg_paths_rel2_sp,
                         pos_paths_rel1_rd, pos_paths_rel2_rd, neg_paths_rel1_rd, neg_paths_rel2_rd
                         )

            return vectors, questions, pos_paths, neg_paths,pos_paths_rel1_sp, pos_paths_rel2_sp,neg_paths_rel1_sp, neg_paths_rel2_sp, \
            pos_paths_rel1_rd, pos_paths_rel2_rd, neg_paths_rel1_rd, neg_paths_rel2_rd



def break_path(path,special_chars):
    '''
        Given a path which always starts with special characters . Give two paths.
    :param path:
    :param special_chars:  a list of special characters
    :return:
    '''

    second_sc_index = None

    for index,p in enumerate(path[1:]):
        if p in special_chars:
            second_sc_index = index + 1
    if second_sc_index:
        path1 = path[:second_sc_index]
        path2 = path[second_sc_index:]
    else:
        path1 = path
        path2 = None

    return path1,path2


def create_dataset_slotpointer(file, max_sequence_length, relations, _dataset, _dataset_specific_data_dir,
                            _model_specific_data_dir
                            , _model='core_chain_pairwise'):
    try:
        raise EOFError
    except (EOFError,IOError) as e:
        with open(os.path.join(_dataset_specific_data_dir % {'dataset': _dataset}, file)) as fp:
            dataset = json.load(fp)
            dataset = dataset

            dummy_path = [0]
            ignored = []

            pos_paths = []
            for i in dataset:
                path_id = i['parsed-data']['path_id']
                positive_path = []
                try:
                    for index,p in enumerate(path_id):
                        positive_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p[0])]
                        positive_path += relations[int(p[1:])][3].tolist()

                except (TypeError, ValueError) as e:
                    ignored.append(i)
                    continue
                pos_paths.append(positive_path)

            '''     
                        $$$$$$$$$$$$$$$$$$$$$$$$$$$
                        $$$$$$$$$$$$$$$$$$$$$$$$$$$

                 The question id is not correct. It sis fucking out of sync 

                        $$$$$$$$$$$$$$$$$$$$$$$$$$$$
                        $$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                        positive_path += relations[int(p[1:])][3].tolist()


                To solve this issue in qald file. Execute scripts/update_questionid_qald file from ipython
                This remaps the question['uri'] to glove(question['uri']). This is a hack, but something clear needs to be worked out.


            '''
            questions = [i['uri']['question-id'] for i in dataset if i not in ignored]
            questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')

            neg_paths = []

            for i in range(0, len(pos_paths)):
                if i in ignored:
                    continue
                datum = dataset[i]
                negative_paths_id = datum['uri']['hop-2-properties'] + datum['uri']['hop-1-properties']
                np.random.shuffle(negative_paths_id)
                negative_paths = []

                for neg_path in negative_paths_id:
                    negative_path = []
                    for p in neg_path:
                        try:
                            negative_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p)]
                        except ValueError:
                            negative_path += relations[int(p)][3].tolist()
                    negative_paths.append(negative_path)
                negative_paths = remove_positive_path(pos_paths[i], negative_paths)

                try:
                    negative_paths = np.random.choice(negative_paths, 1000)


                except ValueError:
                    if len(negative_paths) == 0:
                        try:
                            negative_paths = neg_paths[-1]
                            print("Using previous question's paths for this since no neg paths for this question.")
                        except IndexError:
                            print("at index error. Moving forward due to a hack")
                            negative_paths = np.asarray([1])
                    else:
                        index = np.random.randint(0, len(negative_paths), 1000)
                        negative_paths = np.array(negative_paths)
                        negative_paths = negative_paths[index]
                neg_paths.append(negative_paths)


            pos_paths_rel1 = []
            pos_paths_rel2 = []
            neg_paths_rel1 = []
            neg_paths_rel2 = []
            special_char = [embeddings_interface.vocabularize(['+']),embeddings_interface.vocabularize(['-'])]
            for pps in pos_paths:
                p1,p2 = break_path(pps,special_char)
                pos_paths_rel1.append(p1)
                if p2 is not None:
                    pos_paths_rel2.append(p2)
                else:
                    pos_paths_rel2.append(dummy_path)

            for npps in neg_paths:
                temp_neg_paths_rel1  = []
                temp_neg_paths_rel2 = []
                for npp in npps:
                    p1,p2 = break_path(npp,special_char)
                    temp_neg_paths_rel1.append(p1)
                    if p2 is not None:
                        temp_neg_paths_rel2.append(p2)
                    else:
                        temp_neg_paths_rel2.append(dummy_path)
                neg_paths_rel1.append(temp_neg_paths_rel1)
                neg_paths_rel2.append(temp_neg_paths_rel2)


            for i in range(0, len(neg_paths)):
                neg_paths[i] = pad_sequences(neg_paths[i], maxlen=max_sequence_length, padding='post')
                neg_paths_rel1[i] = pad_sequences(neg_paths_rel1[i], maxlen=max_sequence_length, padding='post')
                neg_paths_rel2[i] = pad_sequences(neg_paths_rel2[i], maxlen=max_sequence_length, padding='post')
            neg_paths = np.asarray(neg_paths)
            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
            pos_paths_rel1 = pad_sequences(pos_paths_rel1, maxlen=max_sequence_length, padding='post')
            pos_paths_rel2 = pad_sequences(pos_paths_rel2, maxlen=max_sequence_length, padding='post')

            vocab, vectors = vocab_master.load()

            # Map everything
            unks_counter = 0
            # number of unks
            for i in range(len(questions)):
                for index in range(len(questions[i])):
                    try:
                        questions[i][index] = vocab[questions[i][index]]
                    except KeyError:
                        unks_counter = unks_counter + 1
                        questions[i][index] = 1
                # questions[i] = np.asarray([vocab[key] for key in questions[i]])
                pos_paths[i] = np.asarray([vocab[key] for key in pos_paths[i]])
                pos_paths_rel1[i] = np.asarray([vocab[key] for key in pos_paths_rel1[i]])
                pos_paths_rel2[i] = np.asarray([vocab[key] for key in pos_paths_rel2[i]])

                for j in range(len(neg_paths[i])):
                    neg_paths[i][j] = np.asarray([vocab[key] for key in neg_paths[i][j]])
                    neg_paths_rel1[i][j] = np.asarray([vocab[key] for key in neg_paths_rel1[i][j]])
                    neg_paths_rel2[i][j] = np.asarray([vocab[key] for key in neg_paths_rel2[i][j]])


            print("places where glove id exists and not in vecotrs ", unks_counter)



            with open(os.path.join(_model_specific_data_dir % {'dataset': _dataset, 'model': _model},
                                   file + ".mapped.npz"), "w+") as data:
                np.savez(data, questions, pos_paths, neg_paths)

            return vectors, questions, pos_paths, neg_paths


def create_dataset_rdf(file, max_sequence_length, _dataset, _dataset_specific_data_dir,
                            _model_specific_data_dir
                            , _model='core_chain_pairwise'):
    with open(os.path.join(_dataset_specific_data_dir % {'dataset': _dataset, 'model': _model},
                           file)) as data:
        dataset = json.load(data)
        # Empty arrays
        questions = []
        pos_paths = []
        neg_paths = []

        for i in range(len(dataset[:int(.80 * len(dataset))])):

            datum = dataset[i]

            '''
                Extracting and padding the positive paths.
            '''
            if '?x' in datum['parsed-data']['constraints'].keys():
                pos_path = "x + " + dbp.get_label(datum['parsed-data']['constraints']['?x'])
            elif '?uri' in datum['parsed-data']['constraints'].keys():
                pos_path = "uri + " + dbp.get_label(datum['parsed-data']['constraints']['?uri'])
            else:
                continue
            pos_path = embeddings_interface.vocabularize(nlutils.tokenize(pos_path))
            pos_paths.append(pos_path)

            # Question
            question = np.zeros((max_sequence_length), dtype=np.int64)
            unpadded_question = np.asarray(datum['uri']['question-id'])
            question[:min(len(unpadded_question), max_sequence_length)] = unpadded_question
            questions.append(question)

            # Negative Path
            unpadded_neg_path = datum["rdf-type-constraints"]
            unpadded_neg_path = remove_positive_path(pos_path, unpadded_neg_path)
            np.random.shuffle(unpadded_neg_path)
            unpadded_neg_path = pad_sequences(unpadded_neg_path, maxlen=max_sequence_length, padding='post')

            '''
                Remove positive path from negative paths.
            '''

            try:
                neg_path = np.random.choice(unpadded_neg_path, 200)
            except ValueError:
                if len(unpadded_neg_path) == 0:
                    neg_path = neg_paths[-1]
                    print("Using previous question's paths for this since no neg paths for this question.")
                else:
                    index = np.random.randint(0, len(unpadded_neg_path), 200)
                    unpadded_neg_path = np.array(unpadded_neg_path)
                    neg_path = unpadded_neg_path[index]

            neg_paths.append(neg_path)

        # Convert things to nparrays
        questions = np.asarray(questions, dtype=np.int64)

        # questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
        pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
        neg_paths = np.asarray(neg_paths)

        vocab, vectors = vocab_master.load()

        # Map everything
        for i in range(len(questions)):
            questions[i] = np.asarray([vocab[key] for key in questions[i]])
            pos_paths[i] = np.asarray([vocab[key] for key in pos_paths[i]])

            for j in range(len(neg_paths[i])):
                neg_paths[i][j] = np.asarray([vocab[key] for key in neg_paths[i][j]])

        return vectors, questions, pos_paths, neg_paths



def create_dataset_runtime(file,_dataset,_dataset_specific_data_dir,split_point=.80):

    '''
        Function loads the data from the _dataset_specific_data_dir+ file and splits it in case of LCQuAD 
    '''

    if _dataset == 'qald':
        id_data_test = json.load(open(os.path.join(_dataset_specific_data_dir , file)))
    elif _dataset == 'lcquad':
        # Load the main data
        id_data = json.load(open(os.path.join(_dataset_specific_data_dir , file)))

        # Split it.
        id_data_test = id_data[int(.80*len(id_data)):]
    else:
        print("warning: Functionality for transfer-a,transfer-b,transfer-c and proper-tranfer-qald is not implemented.")
        id_data_test = []

    vocab, vectors = vocab_master.load()

    return id_data_test, vocab, vectors



def construct_paths(data, relations, gloveid_to_embeddingid, qald=False):
    """
        For a given datanode, the function constructs positive and negative paths and prepares question uri.
        :param data: a data node of id_big_data
        relations : a dictionary which maps relation id to meta inforamtion like surface form, embedding id
        of surface form etc.
        :return: unpadded , continous id spaced question, positive path, negative paths
    """
    question = np.asarray(data['uri']['question-id'])
    # questions = pad_sequences([question], maxlen=max_length, padding='post')

    # inverse id version of positive path and creating a numpy version
    positive_path_id = data['parsed-data']['path_id']
    no_positive_path = False
    if positive_path_id == [-1]:
        positive_path = np.asarray([-1])
        no_positive_path = True
    else:
        positive_path = []
        for path in positive_path_id:
            positive_path += [embeddings_interface.SPECIAL_CHARACTERS.index(path[0])]
            positive_path += relations[int(path[1:])][3].tolist()
        positive_path = np.asarray(positive_path)
    # padded_positive_path = pad_sequences([positive_path], maxlen=max_length, padding='post')

    # negative paths from id to surface form id
    negative_paths_id = data['uri']['hop-2-properties'] + data['uri']['hop-1-properties']
    negative_paths = []
    for neg_path in negative_paths_id:
        negative_path = []
        for path in neg_path:
            try:
                negative_path += [embeddings_interface.SPECIAL_CHARACTERS.index(path)]
            except ValueError:
                negative_path += relations[int(path)][3].tolist()
        negative_paths.append(np.asarray(negative_path))
    negative_paths = np.asarray(negative_paths)
    # negative paths padding
    # padded_negative_paths = pad_sequences(negative_paths, maxlen=max_length, padding='post')

    # explicitly remove any positive path from negative path
    negative_paths = remove_positive_path(positive_path, negative_paths)

    # remap all the id's to the continous id space.

    # passing all the elements through vocab
    question = np.asarray([gloveid_to_embeddingid[key] for key in question])
    if not no_positive_path:
        positive_path = np.asarray([gloveid_to_embeddingid[key] for key in positive_path])
    for i in range(0, len(negative_paths)):
        # temp = []
        for j in xrange(0, len(negative_paths[i])):
            try:
                negative_paths[i][j] = gloveid_to_embeddingid[negative_paths[i][j]]
            except:
                negative_paths[i][j] = gloveid_to_embeddingid[0]
                # negative_paths[i] = np.asarray(temp)
                # negative_paths[i] = np.asarray([vocab[key] for key in negative_paths[i] if key in vocab.keys()])
    if qald:
        return question, positive_path, negative_paths, no_positive_path
    return question, positive_path, negative_paths



class TrainingDataGenerator(Dataset):
    def __init__(self, data, max_length, neg_paths_per_epoch, batch_size,total_negative_samples,pointwise=False,schema='default'):
        self.dummy_y = np.zeros(batch_size)
        self.firstDone = False
        self.max_length = max_length
        self.neg_paths_per_epoch = neg_paths_per_epoch
        self.total_negative_samples = total_negative_samples
        self.pointwise = pointwise
        self.schema = schema

        questions =  data['train_questions']
        pos_paths = data['train_pos_paths']
        neg_paths = data['train_neg_paths']

        if schema == 'slotptr':
            self.pos_paths_rel1 = data['train_pos_paths_rel1_sp']
            self.pos_paths_rel2 = data['train_pos_paths_rel2_sp']
            self.neg_paths_rel1 = data['train_neg_paths_rel1_sp']
            self.neg_paths_rel2 = data['train_neg_paths_rel2_sp']

        elif schema == 'reldet':
            self.pos_paths_rel1 = data['train_pos_paths_rel1_rd']
            self.pos_paths_rel2 = data['train_pos_paths_rel2_rd']
            self.neg_paths_rel1 = data['train_neg_paths_rel1_rd']
            self.neg_paths_rel2 = data['train_neg_paths_rel2_rd']

        # self.questions = np.reshape(np.repeat(np.reshape(questions,
        #                                     (questions.shape[0], 1, questions.shape[1])),
        #                          neg_paths_per_epoch, axis=1), (-1, max_length))
        # print questions.shape
        # self.temp = np.reshape(questions,
        #                   (questions.shape[0], 1, questions.shape[1]))
        # self.temp = np.repeat((self.temp), neg_paths_per_epoch, axis=1)

        self.questions = np.reshape(np.repeat((np.reshape(questions,
                          (questions.shape[0], 1, questions.shape[1]))), neg_paths_per_epoch, axis=1), (-1, max_length))

        self.pos_paths = np.reshape(np.repeat(np.reshape(pos_paths,
                                            (pos_paths.shape[0], 1, pos_paths.shape[1])),
                                 neg_paths_per_epoch, axis=1), (-1, max_length))


        self.neg_paths = neg_paths

        sampling_index = np.random.randint(0, self.total_negative_samples, self.neg_paths_per_epoch)

        self.neg_paths_sampled = np.reshape(self.neg_paths[:,sampling_index, :],
                                            (-1, self.max_length))

        if self.schema != 'default':

            self.neg_paths_rel1_sampled = np.reshape(self.neg_paths_rel1[:,sampling_index, :],
                                            (-1, self.max_length))

            self.neg_paths_rel2_sampled = np.reshape(self.neg_paths_rel2[:, sampling_index, :],
                                                     (-1, self.max_length))

            self.pos_paths_rel1 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel1,
                                                                  (
                                                                  self.pos_paths_rel1.shape[0], 1, self.pos_paths_rel1.shape[1])),
                                                       self.neg_paths_per_epoch, axis=1), (-1, self.max_length))

            self.pos_paths_rel2 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel2,
                                                                  (
                                                                  self.pos_paths_rel2.shape[0], 1, self.pos_paths_rel2.shape[1])),
                                                       self.neg_paths_per_epoch, axis=1), (-1, self.max_length))

        if schema == 'default':

            self.questions_shuffled, self.pos_paths_shuffled, self.neg_paths_shuffled = \
            shuffle(self.questions, self.pos_paths, self.neg_paths_sampled)

        else:

            self.questions_shuffled, self.pos_paths_shuffled, self.pos_paths_rel1_shuffled, self.pos_paths_rel2_shuffled, self.neg_paths_shuffled, self.neg_paths__rel1_shuffled, self.neg_paths__rel2_shuffled= \
                shuffle(self.questions, self.pos_paths, self.pos_paths_rel1, self.pos_paths_rel2 ,self.neg_paths_sampled, self.neg_paths_rel1_sampled, self.neg_paths_rel2_sampled)

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.questions) / self.batch_size)

    def __getitem__(self, idx):
        """
            Called at every iter.

            If code not pointwise, simple sample (not randomly) next batch items.
            If pointwise:
                you use the same sampled things, only that you then concat neg and pos paths,
                and subsample half from there.

        :param idx:
        :return:
        """
        index = lambda x: x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_questions = index(self.questions_shuffled)    # Shape (batch, seqlen)
        batch_pos_paths = index(self.pos_paths_shuffled)    # Shape (batch, seqlen)
        batch_neg_paths = index(self.neg_paths_shuffled)    # Shape (batch, seqlen)
        if self.schema != 'default':
            batch_neg_paths_rel1 = index(self.neg_paths__rel1_shuffled)    # Shape (batch, seqlen)
            batch_neg_paths_rel2 = index(self.neg_paths__rel2_shuffled)    # Shape (batch, seqlen)
            batch_pos_paths_rel1 = index(self.pos_paths_rel1_shuffled)    # Shape (batch, seqlen)
            batch_pos_paths_rel2 = index(self.pos_paths_rel2_shuffled)    # Shape (batch, seqlen)

        if self.pointwise:
            questions = np.vstack((batch_questions, batch_questions))
            paths = np.vstack((batch_pos_paths, batch_neg_paths))
            if self.schema != 'default':
                paths_rel1 = np.vstack((batch_pos_paths_rel1, batch_neg_paths_rel1))
                paths_rel2 = np.vstack((batch_pos_paths_rel2, batch_neg_paths_rel2))

            # Now sample half of thesequestions = np.repeat(batch_questions)
            sample_index = np.random.choice(np.arange(0, 2*self.batch_size), self.batch_size)

            # Y labels are basically decided on whether i \in sample_index > self.batchsize or not.
            y = np.asarray([-1 if index < self.batch_size else 1 for index in sample_index])
            if self.schema == 'default':
                return ([questions[sample_index], paths[sample_index]], y)
            else:
                return ([questions[sample_index], paths[sample_index],paths_rel1[sample_index],paths_rel2[sample_index]], y)

        else:
            if self.schema == 'default':
                return ([batch_questions, batch_pos_paths, batch_neg_paths], self.dummy_y)
            else:
                return ([batch_questions, batch_pos_paths, batch_neg_paths, \
                         batch_pos_paths_rel1,batch_pos_paths_rel2,batch_neg_paths_rel1,batch_neg_paths_rel2], self.dummy_y)

    def shuffle(self):
        """
            To be called at the end of every epoch. We sample new negative paths,
            \and then we shuffle the questions, pos and neg paths in tandem.
        :return: None
        """
        sampling_index = np.random.randint(0, self.total_negative_samples, self.neg_paths_per_epoch)
        self.neg_paths_sampled = np.reshape(
            self.neg_paths[:, sampling_index , :],
            (-1, self.max_length))


        if self.schema != 'default':

            self.neg_paths_rel1_sampled = np.reshape(self.neg_paths_rel1[:,sampling_index, :],
                                            (-1, self.max_length))

            self.neg_paths_rel2_sampled = np.reshape(self.neg_paths_rel2[:, sampling_index, :],
                                                     (-1, self.max_length))

            self.pos_paths_rel1 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel1,
                                                                  (
                                                                  self.pos_paths_rel1.shape[0], 1, self.pos_paths_rel1.shape[1])),
                                                       self.neg_paths_per_epoch, axis=1), (-1, self.max_length))

            self.pos_paths_rel2 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel2,
                                                                  (
                                                                  self.pos_paths_rel2.shape[0], 1, self.pos_paths_rel2.shape[1])),
                                                       self.neg_paths_per_epoch, axis=1), (-1, self.max_length))

            self.questions_shuffled, self.pos_paths_shuffled, self.pos_paths_rel1_shuffled, self.pos_paths_rel2_shuffled, self.neg_paths_shuffled, self.neg_paths__rel1_shuffled, self.neg_paths__rel2_shuffled = \
                shuffle(self.questions, self.pos_paths, self.pos_paths_rel1, self.pos_paths_rel2,
                        self.neg_paths_sampled, self.neg_paths_rel1_sampled, self.neg_paths_rel2_sampled)
        else:
            self.questions_shuffled, self.pos_paths_shuffled, self.neg_paths_shuffled = \
                shuffle(self.questions, self.pos_paths, self.neg_paths_sampled)



class ValidationDataset(Dataset):
    def __init__(self, questions, pos_paths, neg_paths, max_length, neg_paths_per_epoch, batch_size,total_negative_samples):
        self.dummy_y = np.zeros(batch_size)
        self.firstDone = False
        self.max_length = max_length
        self.neg_paths_per_epoch = neg_paths_per_epoch
        self.total_negative_samples = total_negative_samples

        self.questions = np.reshape(np.repeat(np.reshape(questions,
                                            (questions.shape[0], 1, questions.shape[1])),
                                 neg_paths_per_epoch+1, axis=1), (-1, max_length))

        self.pos_paths = np.reshape(pos_paths,
                                            (pos_paths.shape[0], 1, pos_paths.shape[1]))
        self.neg_paths = neg_paths
        neg_paths_sampled = self.neg_paths[:, np.random.randint(0, self.total_negative_samples, self.neg_paths_per_epoch), :]
        self.all_paths = np.reshape(np.concatenate([self.pos_paths, neg_paths_sampled], axis=1), (-1, self.max_length))

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.questions) / self.batch_size)

    def __getitem__(self, idx):
        index = lambda x: x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_questions = index(self.questions)
        batch_all_paths = index(self.all_paths)

        return ([batch_questions, batch_all_paths, np.zeros_like(batch_all_paths)], self.dummy_y)


