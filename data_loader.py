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


config = ConfigParser.RawConfigParser()
config.read('configs/macros.cfg')

SEED = config.getint('Commons', 'seed')
dbp = db_interface.DBPedia(_verbose=True, caching=False)


def load_data(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
              _neg_paths_per_epoch_train,
              _neg_paths_per_epoch_validation, _relations,
              _index, _training_split, _validation_split, _model='core_chain_pairwise',_pairwise=True, _debug=True,_rdf=False):
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
    '''
    _pairwise = True
    if _pairwise:
        if not _rdf:
            vectors, questions, pos_paths, neg_paths = create_dataset_pairwise(file=_file, max_sequence_length=_max_sequence_length, relations=_relations, _dataset=_dataset, _dataset_specific_data_dir=_dataset_specific_data_dir,
            _model_specific_data_dir = _model_specific_data_dir, _model = 'core_chain_pairwise')


        else:
            vectors, questions, pos_paths, neg_paths = create_dataset_rdf(file=_file, max_sequence_length=_max_sequence_length, _dataset=_dataset, _dataset_specific_data_dir=_dataset_specific_data_dir,
            _model_specific_data_dir=_model_specific_data_dir)

    '''
        Making sure that positive path is not the part of negative paths.
    '''
    counter = 0
    for i in range(0, len(pos_paths)):
        temp = -1
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
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

    train_pos_paths = training_split(pos_paths)
    train_neg_paths = training_split(neg_paths)
    train_questions = training_split(questions)

    valid_pos_paths = validation_split(pos_paths)
    valid_neg_paths = validation_split(neg_paths)
    valid_questions = validation_split(questions)

    if not _index:
        test_pos_paths = testing_split(pos_paths)
        test_neg_paths = testing_split(neg_paths)
        test_questions = testing_split(questions)

    dummy_y_train = np.zeros(len(train_questions) * _neg_paths_per_epoch_train)
    dummy_y_valid = np.zeros(len(valid_questions) * (_neg_paths_per_epoch_validation + 1))

    if _debug:
        print(train_questions.shape)
        print(train_pos_paths.shape)
        print(train_neg_paths.shape)

        print(valid_questions.shape)
        print(valid_pos_paths.shape)
        print(valid_neg_paths.shape)

        if not _index:
            print(test_questions.shape)
            print(test_pos_paths.shape)
            print(test_neg_paths.shape)

    if _index:
        print "at _index locations"
        return train_questions, train_pos_paths, train_neg_paths, dummy_y_train, \
               valid_questions, valid_pos_paths, valid_neg_paths, dummy_y_valid, vectors
    else:
        return train_questions, train_pos_paths, train_neg_paths, dummy_y_train, valid_questions, \
               valid_pos_paths, valid_neg_paths, dummy_y_valid, test_questions, test_pos_paths, test_neg_paths, vectors


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
                            , _model='core_chain_pairwise'):
    """
        This file is meant to create data for core-chain ranking ONLY.

    :param file: id_big_data file
    :param max_sequence_length: for padding/cropping
    :param relations: the relations file to backtrack and look up shit.
    :return:
    """

    try:
        with open(os.path.join(_model_specific_data_dir % {'dataset': _dataset, 'model': _model},
                               file + ".mapped.npz")) as data:
            dataset = np.load(data)
            questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
            vocab, vectors = vocab_master.load()
            return vectors, questions, pos_paths, neg_paths
    except (EOFError, IOError) as e:
        with open(os.path.join(_dataset_specific_data_dir % {'dataset': _dataset}, file)) as fp:
            dataset = json.load(fp)

            ignored = []

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

            for i in range(0, len(neg_paths)):
                neg_paths[i] = pad_sequences(neg_paths[i], maxlen=max_sequence_length, padding='post')
            neg_paths = np.asarray(neg_paths)
            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')

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

                for j in range(len(neg_paths[i])):
                    neg_paths[i][j] = np.asarray([vocab[key] for key in neg_paths[i][j]])

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



class TrainingDataGenerator(Dataset):
    def __init__(self, questions, pos_paths, neg_paths, max_length, neg_paths_per_epoch, batch_size,total_negative_samples,pointwise=False):
        self.dummy_y = np.zeros(batch_size)
        self.firstDone = False
        self.max_length = max_length
        self.neg_paths_per_epoch = neg_paths_per_epoch
        self.total_negative_samples = total_negative_samples
        self.pointwise = pointwise

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

        self.neg_paths_sampled = np.reshape(self.neg_paths[:,np.random.randint(0, self.total_negative_samples, self.neg_paths_per_epoch), :],
                                            (-1, self.max_length))

        self.questions_shuffled, self.pos_paths_shuffled, self.neg_paths_shuffled = \
            shuffle(self.questions, self.pos_paths, self.neg_paths_sampled)

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

        if self.pointwise:
            questions = np.vstack((batch_questions, batch_questions))
            paths = np.vstack((batch_pos_paths, batch_neg_paths))

            # Now sample half of thesequestions = np.repeat(batch_questions)
            sample_index = np.random.choice(np.arange(0, 2*self.batch_size), self.batch_size)

            # Y labels are basically decided on whether i \in sample_index > self.batchsize or not.
            y = np.asarray([-1 if index < self.batch_size else 1 for index in sample_index])
            return ([questions[sample_index], paths[sample_index]], y)

        else:
            return ([batch_questions, batch_pos_paths, batch_neg_paths], self.dummy_y)

    def shuffle(self):
        """
            To be called at the end of every epoch. We sample new negative paths,
            \and then we shuffle the questions, pos and neg paths in tandem.
        :return: None
        """
        self.neg_paths_sampled = np.reshape(
            self.neg_paths[:, np.random.randint(0, self.total_negative_samples, self.neg_paths_per_epoch), :],
            (-1, self.max_length))

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
