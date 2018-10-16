# Shared Feature Extraction Layer
'''

    DELETE
    DELETE
    DELETE
    DELETE
    DELETE
    DELETE
    DELETE
    DELETE

'''
from __future__ import absolute_import
import os
import sys
import json

import numpy as np



import network as n
import network_corechain as n_cc
from utils import embeddings_interface
from utils import dbpedia_interface as db_interface
from utils import natural_language_utilities as nlutils
from utils import prepare_vocab_continous as vocab_master


# Some Macros
DEBUG = True
# LCQUAD = True
# DATA_DIR = './data/models/rdf/lcquad/' if LCQUAD else './data/models/rdf/qald/'
# RESOURCE_DIR = './resources_v8/rdf'
# ID_DIR = './resources_v8'
EPOCHS = 300
BATCH_SIZE = 180 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
NEGATIVE_SAMPLES = 270

# Set up directories
n.NEGATIVE_SAMPLES = 200
n.BATCH_SIZE = BATCH_SIZE

dbp = db_interface.DBPedia(_verbose=True, caching=False)
embeddings_interface.__check_prepared__()

def create_dataset():
    """
        Prepares the training data to be **directly** fed into the model.

    :param file:
    :param max_sequence_length:
    :return:
    """

    try:

        # @TODO THIS BLOCK
        # with open(os.path.join(RESOURCE_DIR, file + ".mapped.npz")) as data, open(os.path.join('./resources_v8', file + ".vocab.pickle")) as idx:
        #     dataset = np.load(data)
        #     # dataset = dataset[:10]
        #     questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
        #     vocab = pickle.load(idx)
        #     vectors = glove_embeddings[vocab.keys()]
        #     return vectors, questions, pos_paths, neg_paths
        raise EOFError

    except (EOFError, IOError) as e:
        with open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset':n.DATASET}, FILENAME)) as fp:
            dataset = json.load(fp)
            # dataset = dataset[:10]

            # Empty arrays
            questions = []
            pos_paths = []
            neg_paths = []

            for i in range(len(dataset[:int(.80*len(dataset))])):

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
                question = np.zeros((n.MAX_SEQ_LENGTH), dtype=np.int64)
                unpadded_question = np.asarray(datum['uri']['question-id'])
                question[:min(len(unpadded_question), n.MAX_SEQ_LENGTH)] = unpadded_question
                questions.append(question)

                # Negative Path
                unpadded_neg_path = datum["rdf-type-constraints"]
                unpadded_neg_path = n.remove_positive_path(pos_path, unpadded_neg_path)
                np.random.shuffle(unpadded_neg_path)
                unpadded_neg_path = nlutils.pad_sequence(unpadded_neg_path, max_length=n.MAX_SEQ_LENGTH)

                '''
                    Remove positive path from negative paths.
                '''

                try:
                    neg_path = np.random.choice(unpadded_neg_path,200)
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
            pos_paths = nlutils.pad_sequence(pos_paths, max_length=n.MAX_SEQ_LENGTH)
            neg_paths = np.asarray(neg_paths)

            # '''
            #     Take care of vocabulary.
            #     Note: if vocabulary changed, all the models are rendered useless.
            # '''
            # try:
            #     vocab = pickle.load(open('resources_v8/id_big_data.json.vocab.pickle'))
            # except (IOError, EOFError) as e:
            #     vocab = {}
            #
            # if DEBUG:
            #     print(questions.shape)
            #     print(pos_paths.shape)
            #     print(neg_paths.shape)
            #
            # all = np.concatenate([questions, pos_paths, neg_paths.reshape(neg_paths.shape[0]*neg_paths.shape[1],neg_paths.shape[2])], axis=0)
            # uniques = np.unique(all)
            #
            #
            # # ############################################################
            # # Map to new ID space all those which are not a part of vocab#
            # # ############################################################
            #
            # index = len(vocab)
            #
            # for key in uniques:
            #     try:
            #         temp = vocab[key]
            #     except KeyError:
            #     # if key not in vocab.keys():
            #         vocab[key] = index
            #         index += 1
            #
            #
            # # Create slimmer, better, faster, vectors file.
            # vectors = glove_embeddings[uniques]

            vocab, vectors = vocab_master.load()

            # Map everything
            for i in range(len(questions)):
                questions[i] = np.asarray([vocab[key] for key in questions[i]])
                pos_paths[i] = np.asarray([vocab[key] for key in pos_paths[i]])

                for j in range(len(neg_paths[i])):
                    neg_paths[i][j] = np.asarray([vocab[key] for key in neg_paths[i][j]])

            # with open(os.path.join(n.MODEL_SPECIFIC_DATA_DIR % {'model':'rdf', 'dataset':n.DATASET}, file + ".mapped.npz"), "w+") as data:
            #     np.savez(data, questions, pos_paths, neg_paths)
                # pickle.dump(vocab,idx)

            return vectors, questions, pos_paths, neg_paths


if __name__ == "__main__":

    gpu = sys.argv[1]
    DATASET = sys.argv[2].strip()

    # See if the args are valid.
    while True:
        try:
            assert gpu in ['0', '1', '2', '3']
            assert DATASET in ['lcquad', 'qald']
            break
        except AssertionError:
            gpu = input("Did not understand which gpu to use. Please write it again: ")
            DATASET = input("Did not understand which Dataset to use. Please write it again: ")

    relations = n.load_relation()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    FILENAME = ['qald_id_big_data_train.json', 'qald_id_big_data_test.json'] if DATASET == 'qald' else 'id_big_data.json'
    n.MODEL = 'rdf'
    n.DATASET = DATASET

    # If we're working with QALD, we need to open both files, combine, store and give THAT as the filename to createdata
    if DATASET == 'qald':

        # id_train = json.load(open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset': DATASET}, FILENAME[0])))
        # id_test = json.load(open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset': DATASET}, FILENAME[1])))
        #
        # index = len(id_train) - 1
        # FILENAME = 'combined_qald.json'
        # json.dump(id_train + id_test,
        #           open(os.path.join(n.DATASET_SPECIFIC_DATA_DIR % {'dataset': DATASET}, FILENAME), 'w+'))

        FILENAME_train, FILENAME_test = FILENAME[0], FILENAME[1]

        # Compute train data
        FILENAME = FILENAME_train
        vectors, questions_train, pos_paths_train, neg_paths_train = create_dataset()

        # Compute test data
        FILENAME = FILENAME_test
        _, questions_test, pos_paths_test, neg_paths_test = create_dataset()

        index = len(questions_train) - 1
        questions = np.concatenate((questions_train, questions_test))
        pos_paths = np.concatenate((pos_paths_train, pos_paths_test))
        neg_paths = np.concatenate((neg_paths_train, neg_paths_test))

    else:
        index = None
        vectors, questions, pos_paths, neg_paths = create_dataset()
    # n.BATCH_SIZE = BATCH_SIZE
    n_cc.bidirectional_dot(gpu, vectors, questions, pos_paths, neg_paths, 10, 200, _index=index,rdf=True)
