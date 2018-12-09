'''
    Loads data from the folder and creates a test/train/validation splits.
'''

import os
import sys
import json
import math
import pickle
import traceback
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
# from utils.natural_language_utilities import pad_sequence

if sys.version_info[0] == 3:
    import configparser as ConfigParser
else:
    import ConfigParser

# Custom imports
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils
import auxiliary as aux


sys.path.append('/data/priyansh/conda/fastai')
import os
os.environ['QT_QPA_PLATFORM']='offscreen'


config = ConfigParser.RawConfigParser()
config.read('configs/macros.cfg')

SEED = config.getint('Commons', 'seed')
dbp = db_interface.DBPedia(_verbose=True, caching=False)
COMMON_DATA_DIR = 'data/data/common'

embeddings_interface.__check_prepared__()
vocabularize_relation_old = lambda path: embeddings_interface.vocabularize(nlutils.tokenize(dbp.get_label(path))).tolist()
special_char_vec = [vocabularize_relation_old(i)for i in embeddings_interface.SPECIAL_CHARACTERS]
vocabularize_relation = lambda path: special_char_vec[embeddings_interface.SPECIAL_CHARACTERS.index(path)]

rel_dict = pickle.load(open('data/data/common/relations.pickle','rb'))
# inv_rel_dict = {v[0]: k for k, v in rel_dict.items()}
inv_rel_dict = {v[0]: [k]+v[1:] for k, v in rel_dict.items()}

def load_data(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
              _neg_paths_per_epoch_train,
              _neg_paths_per_epoch_validation, _relations,
              _index, _training_split, _validation_split, _model='core_chain_pairwise',_pairwise=True, _debug=True,_rdf=False,
              _schema = 'default',k=-1):
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
            _model_specific_data_dir = _model_specific_data_dir, _model = 'core_chain_pairwise',k=k)


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
                    if not _rdf:
                        neg_paths_rel1_sp[i][j] = neg_paths_rel1_sp[i][j + 10]
                        neg_paths_rel2_sp[i][j] = neg_paths_rel2_sp[i][j + 10]
                        neg_paths_rel1_rd[i][j] = neg_paths_rel1_rd[i][j + 10]
                        neg_paths_rel2_rd[i][j] = neg_paths_rel2_rd[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
                    if not _rdf:
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


    '''
        DELETE
        DELETE
        DELETE
        DELETE
        DELETE
    '''
    entity = questions
    data['train_pos_paths'] = training_split(pos_paths)
    if not _rdf:
        data['train_pos_paths_rel1_sp'] = training_split(pos_paths_rel1_sp)
        data['train_pos_paths_rel2_sp'] = training_split(pos_paths_rel2_sp)
        data['train_pos_paths_rel1_rd'] = training_split(pos_paths_rel1_rd)
        data['train_pos_paths_rel2_rd'] = training_split(pos_paths_rel2_rd)

    data['train_neg_paths'] = training_split(neg_paths)
    if not _rdf:
        data['train_neg_paths_rel1_sp'] = training_split(neg_paths_rel1_sp)
        data['train_neg_paths_rel2_sp']= training_split(neg_paths_rel2_sp)
        data['train_neg_paths_rel1_rd'] = training_split(neg_paths_rel1_rd)
        data['train_neg_paths_rel2_rd'] = training_split(neg_paths_rel2_rd)

    data['train_questions'] = training_split(questions)

    if not _rdf:
        data['train_entity'] = training_split(entity)
    data['valid_pos_paths'] = validation_split(pos_paths)
    if not _rdf:
        data['valid_pos_paths_rel1_sp'] = validation_split(pos_paths_rel1_sp)
        data['valid_pos_paths_rel2_sp'] = validation_split(pos_paths_rel2_sp)
        data['valid_pos_paths_rel1_rd'] = validation_split(pos_paths_rel1_rd)
        data['valid_pos_paths_rel2_rd'] = validation_split(pos_paths_rel2_rd)

    data['valid_neg_paths'] = validation_split(neg_paths)
    if not _rdf:
        data['valid_neg_paths_rel1_sp'] = validation_split(neg_paths_rel1_sp)
        data['valid_neg_paths_rel2_sp'] = validation_split(neg_paths_rel2_sp)
        data['valid_neg_paths_rel1_rd'] = validation_split(neg_paths_rel1_rd)
        data['valid_neg_paths_rel2_rd'] = validation_split(neg_paths_rel2_rd)


    data['valid_questions'] = validation_split(questions)
    if not _rdf:
       data['valid_entity'] = validation_split(entity)

    if not _index:
        data['test_pos_paths'] = testing_split(pos_paths)
        if not _rdf:
            data['test_pos_paths_rel1_sp'] = testing_split(pos_paths_rel1_sp)
            data['test_pos_paths_rel2_sp'] = testing_split(pos_paths_rel2_sp)
            data['test_pos_paths_rel1_rd'] = testing_split(pos_paths_rel1_rd)
            data['test_pos_paths_rel2_rd'] = testing_split(pos_paths_rel2_rd)


        data['test_neg_paths'] = testing_split(neg_paths)
        if not _rdf:
            data['test_neg_paths_rel1_sp'] = testing_split(neg_paths_rel1_sp)
            data['test_neg_paths_rel2_sp'] = testing_split(neg_paths_rel2_sp)
            data['test_neg_paths_rel1_rd'] = testing_split(neg_paths_rel1_rd)
            data['test_neg_paths_rel2_rd'] = testing_split(neg_paths_rel2_rd)

        data['test_questions'] = testing_split(questions)
        if not _rdf:
            data['test_entity'] = testing_split(entity)


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


def relation_table_lookup(lookup_str, glove_id_sf_to_glove_id_rel):
    '''
        given an np array of durface form of relation it gives a glove id for the whole relation string(list)
    '''
    if list(lookup_str)[0] == 0:
        return None

    key = str(list(lookup_str)[1:]) # 1 ownwards because first is the sign
    return glove_id_sf_to_glove_id_rel[key]


def relation_table_lookup_reverse_legacy(lookup_str, glove_id_sf_to_glove_id_rel,
                                         embeddingid_to_gloveid, gloveid_to_embeddingid):
    """
        given an np array of durface form of relation it gives a glove id for the whole relation string(list)
    """
    if list(lookup_str)[0] == 0:
        print(lookup_str)
        return None

    lookup_str = [embeddingid_to_gloveid[i] for i in lookup_str]
    key = str(list(lookup_str)[1:])  # 1 onwards because first is the sign
    a = glove_id_sf_to_glove_id_rel[key]
    return [gloveid_to_embeddingid[a[0]]]


def relation_table_lookup_reverse(lookup_str, glove_id_sf_to_glove_id_rel):
    """
        given an np array of surface form of relation it gives a glove id for the whole relation string(list)
    """
    if list(lookup_str)[0] == 0:
        print(lookup_str)
        return None

    # lookup_str = [embeddingid_to_gloveid[i] for i in lookup_str]
    key = str(list(lookup_str)[1:])  # 1 ownwards because first is the sign
    a = glove_id_sf_to_glove_id_rel[key]
    return a[0]


def create_relation_lookup_table(COMMON_DATA_DIR):
    """

        Creates a lookup table with key being
            glove id of the surface form of relation (str(list(numpy))) and the whole relation id
    :param COMMON_DATA_DIR:
    :return:
    """

    inv_relation = aux.load_inverse_relation(COMMON_DATA_DIR)
    glove_id_sf_to_glove_id_rel = {}
    for keys in inv_relation:
        k = str(list(inv_relation[keys][3]))
        #THIS MUST BE REPLACED REPLACE REPLACE REPALCE
        '''
            REPLACE
            REPLACE
            REPLACE
            REPLACE
            REPLACE
        '''
        glove_id_sf_to_glove_id_rel[k] = list(inv_relation[keys][-1])

    return glove_id_sf_to_glove_id_rel


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
                            _model_specific_data_dir, _model='core_chain_pairwise',k=-1):
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
        # raise EOFError
        with open(os.path.join(_model_specific_data_dir % {'dataset': _dataset, 'model': _model},
                               file + ".mapped.npz"),'rb') as data:
            dataset = np.load(data)
            questions, pos_paths, neg_paths, \
            pos_paths_rel1_sp, pos_paths_rel2_sp,neg_paths_rel1_sp, neg_paths_rel2_sp, \
            pos_paths_rel1_rd, pos_paths_rel2_rd, neg_paths_rel1_rd, neg_paths_rel2_rd = dataset['arr_0'][:k],\
                                                                                         dataset['arr_1'][:k],\
                                                                                         dataset['arr_2'][:k], \
                                                                                         dataset['arr_3'][:k],\
                                                                                         dataset['arr_4'][:k],\
                                                                                         dataset['arr_5'][:k], \
                                                                                         dataset['arr_6'][:k], \
                                                                                         dataset['arr_7'][:k],\
                                                                                         dataset['arr_8'][:k],\
                                                                                         dataset['arr_9'][:k],\
                                                                                         dataset['arr_10'][:k]



            # vocab, vectors = vocab_master.load_ulmfit()
            # vocab, vectors = vocab_master.load()
            vectors = embeddings_interface.vectors
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
                #@TODO: Change this path as soon as training thing changes.
                path_id = i['parsed-data']['path']
                positive_path = []
                try:
                    for p in path_id:
                        if p in ['+','-']:
                            positive_path += vocabularize_relation(p)
                        else:
                            positive_path += relations[int(p)][3].tolist()
                except (TypeError, ValueError) as e:
                    ignored.append(i)
                    print("error here")
                    print(traceback.print_exc())
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
            questions = nlutils.pad_sequence(questions,max_sequence_length)

            # entity = [i['uri']['entity-id'] for i in dataset if i not in ignored]
            # entity = nlutils.pad_sequence(entity,max_sequence_length)

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
                        if p in embeddings_interface.SPECIAL_CHARACTERS:
                            negative_path += vocabularize_relation(p)
                        else:
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
                    try:
                        temp_neg_paths_rel1_rd.append(relation_table_lookup(p1,glove_id_sf_to_glove_id_rel))
                    except:
                        print()
                        print(p1,p2,i)
                    if p2 is not None:
                        temp_neg_paths_rel2_sp.append(p2)
                        try:
                            temp_neg_paths_rel2_rd.append(relation_table_lookup(p2, glove_id_sf_to_glove_id_rel))
                        except:
                            print(p1, p2, i)
                    else:
                        temp_neg_paths_rel2_sp.append(dummy_path)
                        temp_neg_paths_rel2_rd.append(dummy_path)
                neg_paths_rel1_sp.append(temp_neg_paths_rel1_sp)
                neg_paths_rel2_sp.append(temp_neg_paths_rel2_sp)
                neg_paths_rel1_rd.append(temp_neg_paths_rel1_rd)
                neg_paths_rel2_rd.append(temp_neg_paths_rel2_rd)


            for i in range(0, len(neg_paths)):
                neg_paths[i] = nlutils.pad_sequence(neg_paths[i],max_sequence_length)
                neg_paths_rel1_sp[i] = nlutils.pad_sequence(neg_paths_rel1_sp[i],max_sequence_length)
                neg_paths_rel2_sp[i] = nlutils.pad_sequence(neg_paths_rel2_sp[i],max_sequence_length)
                neg_paths_rel1_rd[i] = nlutils.pad_sequence(neg_paths_rel1_rd[i],max_sequence_length)
                neg_paths_rel2_rd[i] = nlutils.pad_sequence(neg_paths_rel2_rd[i],max_sequence_length)
            neg_paths = np.asarray(neg_paths)
            neg_paths_rel1_sp = np.asarray(neg_paths_rel1_sp)
            neg_paths_rel2_sp = np.asarray(neg_paths_rel2_sp)
            neg_paths_rel1_rd = np.asarray(neg_paths_rel1_rd)
            neg_paths_rel2_rd = np.asarray(neg_paths_rel2_rd)

            pos_paths = nlutils.pad_sequence(pos_paths,max_sequence_length)
            pos_paths_rel1_sp = nlutils.pad_sequence(pos_paths_rel1_sp,max_sequence_length)
            pos_paths_rel2_sp = nlutils.pad_sequence(pos_paths_rel2_sp,max_sequence_length)
            pos_paths_rel1_rd = nlutils.pad_sequence(pos_paths_rel1_rd,max_sequence_length)
            pos_paths_rel2_rd = nlutils.pad_sequence(pos_paths_rel2_rd,max_sequence_length)


            # vocab, vectors = vocab_master.load_ulmfit()
            # vocab, vectors = vocab_master.load()
            vectors = embeddings_interface.vectors

            #Legacy stuff.
            # # Map everything
            # unks_counter = 0
            # # number of unks
            # for i in range(len(questions)):
            #     for index in range(len(questions[i])):
            #         try:
            #             questions[i][index] = vocab[questions[i][index]]
            #         except KeyError:
            #             unks_counter = unks_counter + 1
            #             questions[i][index] = 1
            #     # questions[i] = np.asarray([vocab[key] for key in questions[i]])
            #     pos_paths[i] = np.asarray([vocab[key] for key in pos_paths[i]])
            #     pos_paths_rel1_sp[i] = np.asarray([vocab[key] for key in pos_paths_rel1_sp[i]])
            #     pos_paths_rel2_sp[i] = np.asarray([vocab[key] for key in pos_paths_rel2_sp[i]])
            #     pos_paths_rel1_rd[i] = np.asarray([vocab[key] for key in pos_paths_rel1_rd[i]])
            #     pos_paths_rel2_rd[i] = np.asarray([vocab[key] for key in pos_paths_rel2_rd[i]])
            #
            #     for j in range(len(neg_paths[i])):
            #         neg_paths[i][j] = np.asarray([vocab[key] for key in neg_paths[i][j]])
            #         neg_paths_rel1_sp[i][j] = np.asarray([vocab[key] for key in neg_paths_rel1_sp[i][j]])
            #         neg_paths_rel2_sp[i][j] = np.asarray([vocab[key] for key in neg_paths_rel2_sp[i][j]])
            #         neg_paths_rel1_rd[i][j] = np.asarray([vocab[key] for key in neg_paths_rel1_rd[i][j]])
            #         neg_paths_rel2_rd[i][j] = np.asarray([vocab[key] for key in neg_paths_rel2_rd[i][j]])
            #
            # print("places where glove id exists and not in vecotrs ", unks_counter)
            # return vectors, questions, pos_paths, neg_paths, pos_paths_rel1_sp, pos_paths_rel2_sp, neg_paths_rel1_sp, neg_paths_rel2_sp, \
            #        pos_paths_rel1_rd, pos_paths_rel2_rd, neg_paths_rel1_rd, neg_paths_rel2_rd

            print("att ht place where things are made")
            with open(os.path.join(_model_specific_data_dir % {'dataset': _dataset, 'model': _model},
                                   file + ".mapped.npz"), "wb+") as data:
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


def create_dataset_rdf(file, max_sequence_length, _dataset, _dataset_specific_data_dir,
                            _model_specific_data_dir, _model='core_chain_pairwise',_coomon_dir='data/data/common'):
    with open(os.path.join(_dataset_specific_data_dir % {'dataset': _dataset, 'model': _model},
                           file)) as data:
        dataset = json.load(data)
        # Empty arrays
        questions = []
        pos_paths = []
        neg_paths = []

        for i in range(len(dataset[:int(len(dataset))])):

            datum = dataset[i]

            '''
                Extracting and padding the positive paths.
            '''
            if '?x' in datum['parsed-data']['constraints'].keys():
                pos_path = " ".join(['x',dbp.get_label(datum['parsed-data']['constraints']['?x'])])
                # pos_path = "x " + dbp.get_label(datum['parsed-data']['constraints']['?x'])
            elif '?uri' in datum['parsed-data']['constraints'].keys():
                pos_path = " ".join(['uri', dbp.get_label(datum['parsed-data']['constraints']['?uri'])])
                # pos_path = "uri " + dbp.get_label(datum['parsed-data']['constraints']['?uri'])
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
            for i_neg, path in enumerate(unpadded_neg_path):
                # print(unpadded_neg_path)
                assert len(path) == 2
                # print([path[0]] , inv_rel_dict[path[1]][3],inv_rel_dict[path[1]],path[1])
                path = [path[0]] + inv_rel_dict[path[1]][3].tolist()
                unpadded_neg_path[i_neg] = path
            np.random.shuffle(unpadded_neg_path)
            unpadded_neg_path = nlutils.pad_sequence(unpadded_neg_path,max_sequence_length)

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
        pos_paths = nlutils.pad_sequence(pos_paths, max_sequence_length)
        neg_paths = np.asarray(neg_paths)

        vectors = embeddings_interface.vectors

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

    vectors = embeddings_interface.vectors

    return id_data_test, vectors



def construct_paths(data, relations, qald=False):
    """
        For a given datanode, the function constructs positive and negative paths and prepares question uri.
        :param data: a data node of id_big_data
        relations : a dictionary which maps relation id to meta inforamtion like surface form, embedding id
        of surface form etc.
        :return: unpadded , continous id spaced question, positive path, negative paths
    """
    question = np.asarray(data['uri']['question-id'])
    # entity = np.asarray(data['uri']['entity-id'])
    # questions = pad_sequences([question], maxlen=max_length, padding='post')

    # inverse id version of positive path and creating a numpy version
    positive_path_id = data['parsed-data']['path']   #change this
    no_positive_path = False
    if positive_path_id == -1 or positive_path_id == [-1]:
        positive_path = np.asarray([-1])
        no_positive_path = True
    else:
        positive_path = []
        for p in positive_path_id:
            if p in ['+', '-']:
                positive_path += vocabularize_relation(p)
            else:
                positive_path += relations[int(p)][3].tolist()
        positive_path = np.asarray(positive_path)
    # padded_positive_path = pad_sequences([positive_path], maxlen=max_length, padding='post')

    # negative paths from id to surface form id
    negative_paths_id = data['uri']['hop-2-properties'] + data['uri']['hop-1-properties']
    negative_paths = []
    for neg_path in negative_paths_id:
        negative_path = []
        for p in neg_path:
            if p in embeddings_interface.SPECIAL_CHARACTERS:
                negative_path += vocabularize_relation(p)
            else:
                negative_path += relations[int(p)][3].tolist()
        negative_paths.append(np.asarray(negative_path))
    negative_paths = np.asarray(negative_paths)
    # negative paths padding
    # padded_negative_paths = pad_sequences(negative_paths, maxlen=max_length, padding='post')

    # explicitly remove any positive path from negative path
    negative_paths = remove_positive_path(positive_path, negative_paths)

    # remap all the id's to the continous id space.

    # passing all the elements through vocab
    # question = np.asarray([gloveid_to_embeddingid[key] for key in question])
    # if not no_positive_path:
    #     positive_path = np.asarray([gloveid_to_embeddingid[key] for key in positive_path])
    # for i in range(0, len(negative_paths)):
    #     # temp = []
    #     for j in range(0, len(negative_paths[i])):
    #         try:0.65
    #             negative_paths[i][j] = gloveid_to_embeddingid[negative_paths[i][j]]
    #         except:
    #             negative_paths[i][j] = gloveid_to_embeddingid[0]
                # negative_paths[i] = np.asarray(temp)
                # negative_paths[i] = np.asarray([vocab[key] for key in negative_paths[i] if key in vocab.keys()])
    if qald:
        return question, positive_path, negative_paths, no_positive_path
    return question, positive_path, negative_paths

class TrainingDataGenerator(Dataset):
    def __init__(self, data, max_length, neg_paths_per_epoch, batch_size,total_negative_samples,pointwise=False,schema='default',snip=800):
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
        print(questions.shape)
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
            self.batch_size = batch_questions.shape[0]
            questions = np.vstack((batch_questions, batch_questions))
            paths = np.vstack((batch_pos_paths, batch_neg_paths))
            if self.schema != 'default':
                paths_rel1 = np.vstack((batch_pos_paths_rel1, batch_neg_paths_rel1))
                paths_rel2 = np.vstack((batch_pos_paths_rel2, batch_neg_paths_rel2))

            # Now sample half of thesequestions = np.repeat(batch_questions)
            sample_index = np.random.choice(np.arange(0, 2*self.batch_size), self.batch_size)

            # Y labels are basically decided on whether i \in sample_index > self.batchsize or not.
            y = np.asarray([1 if index < self.batch_size else 0 for index in sample_index])
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
            #
            # self.pos_paths_rel1 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel1,
            #                                                       (
            #                                                       self.pos_paths_rel1.shape[0], 1, self.pos_paths_rel1.shape[1])),
            #                                            self.neg_paths_per_epoch, axis=1), (-1, self.max_length))
            #
            # self.pos_paths_rel2 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel2,
            #                                                       (
            #                                                       self.pos_paths_rel2.shape[0], 1, self.pos_paths_rel2.shape[1])),
            #                                            self.neg_paths_per_epoch, axis=1), (-1, self.max_length))

            self.questions_shuffled, self.pos_paths_shuffled, self.pos_paths_rel1_shuffled,\
            self.pos_paths_rel2_shuffled, self.neg_paths_shuffled, self.neg_paths__rel1_shuffled, self.neg_paths__rel2_shuffled = \
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


