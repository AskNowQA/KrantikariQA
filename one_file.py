'''
    This script will run the whole pipeline at testing time. Following model would be used
        > Core chain ranking model - ./data/training/pairwise/model_30/model.h5
        > Classifier for ask and count. - still need to be trained
        > rdf type classifier - still need to be trained
'''

from __future__ import absolute_import

import os
from utils import embeddings_interface
from utils import natural_language_utilities as nlutils

gpu = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import sys
import json
import math
import pickle
import numpy as np
import pandas as pd
from keras import optimizers
from progressbar import ProgressBar
from utils import embeddings_interface
import data_preparation_rdf_type as drt
import keras.backend.tensorflow_backend as K
from utils import dbpedia_interface as db_interface
from keras.preprocessing.sequence import pad_sequences
from utils import natural_language_utilities as nlutils


# Some Macros
DEBUG = True
DATA_DIR = './data/training/pairwise'
EPOCHS = 300
BATCH_SIZE = 880 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
NEGATIVE_SAMPLES = 1000
OPTIMIZER = optimizers.Adam(LEARNING_RATE)
LOC_RDF_TYPE_LOOKUP = 'resources_v8/rdf_type_lookup.json'

relations = pickle.load(open('resources_v8/relations.pickle'))
dbp = db_interface.DBPedia(_verbose=True, caching=False)

reverse_relations = {}
for keys in relations:
    reverse_relations[relations[keys][0]] = [keys] + relations[keys][1:]


def custom_loss(y_true, y_pred):
    '''
        max margin loss
    '''
    # y_pos = y_pred[0]
    # y_neg= y_pred[1]
    diff = y_pred[:,-1]
    # return K.sum(K.maximum(1.0 - diff, 0.))
    return K.sum(diff)

def rank_precision(model, test_questions, test_pos_paths, test_neg_paths, neg_paths_per_epoch=100, batch_size=1000):
    max_length = test_questions.shape[-1]
    questions = np.reshape(np.repeat(np.reshape(test_questions,
                                            (test_questions.shape[0], 1, test_questions.shape[1])),
                                 neg_paths_per_epoch+1, axis=1), (-1, max_length))
    pos_paths = np.reshape(test_pos_paths,
                                    (test_pos_paths.shape[0], 1, test_pos_paths.shape[1]))
    neg_paths = test_neg_paths[:, np.random.randint(0, NEGATIVE_SAMPLES, neg_paths_per_epoch), :]
    all_paths = np.reshape(np.concatenate([pos_paths, neg_paths], axis=1), (-1, max_length))

    outputs = model.predict([questions, all_paths, np.zeros_like(all_paths)], batch_size=batch_size)[:,0]
    outputs = np.reshape(outputs, (test_questions.shape[0], neg_paths_per_epoch+1))

    precision = float(len(np.where(np.argmax(outputs, axis=1)==0)[0]))/outputs.shape[0]
    return precision

def rank_precision_runtime(model, id_q, id_tp, id_fps, batch_size=1000, max_length=50):
    '''
        A function to pad the data for the model, run model.predict on it and get the resuts.

    :param id_q: A 1D array of the question
    :param id_tp: A 1D array of the true path
    :param id_fps: A list of 1D arrays of false paths
    :param batch_size: int: the batch size the model expects
    :param max_length: int: size with which we pad the data
    :return: ?? @TODO
    '''

    # Create empty matrices
    question = np.zeros((len(id_fps)+1, max_length))
    paths = np.zeros((len(id_fps)+1, max_length))

    # Fill them in
    question[:,:id_q.shape[0]] = np.repeat(id_q[np.newaxis,:min(id_q.shape[0], question.shape[1])],
                                           question.shape[0], axis=0)
    paths[0, :id_tp.shape[0]] = id_tp
    for i in range(len(id_fps)):
        if len(id_fps[i]) > max_length:
            paths[i+1,:min(id_fps[i].shape[0],question.shape[1])] = id_fps[i][:max_length]
        else:
            paths[i+1,:min(id_fps[i].shape[0], question.shape[1])] = id_fps[i]
    # Pass em through the model
    results = model.predict([question, paths, np.zeros_like(paths)], batch_size=batch_size)[:,0]
    return results

def rank_precision_metric(neg_paths_per_epoch):
    def metric(y_true, y_pred):
        scores = y_pred[:, 0]
        scores = K.reshape(scores, (-1, neg_paths_per_epoch+1))
        hits = K.cast(K.shape(K.tf.where(K.tf.equal(K.tf.argmax(scores, axis=1),0)))[0], 'float32')
        precision = hits/K.cast(K.shape(scores)[0], 'float32')
        # precision = float(len(np.where(np.argmax(all_outputs, axis=1)==0)[0]))/all_outputs.shape[0]
        return precision
    return metric

def get_glove_embeddings():
    from utils.embeddings_interface import __check_prepared__
    __check_prepared__('glove')

    from utils.embeddings_interface import glove_embeddings
    return glove_embeddings

def cross_correlation(x):
    a, b = x
    tf = K.tf
    a_fft = tf.fft(tf.complex(a, 0.0))
    b_fft = tf.fft(tf.complex(b, 0.0))
    ifft = tf.ifft(tf.conj(a_fft) * b_fft)
    return tf.cast(tf.real(ifft), 'float32')



def load_data(file, max_sequence_length, relations):
    glove_embeddings = get_glove_embeddings()

    try:
        with open(os.path.join(RESOURCE_DIR, file + ".mapped.npz")) as data, open(os.path.join(RESOURCE_DIR, file + ".index.npy")) as idx:
            dataset = np.load(data)
            # dataset = dataset[:10]
            questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
            index = np.load(idx)
            vectors = glove_embeddings[index]
            return vectors, questions, pos_paths, neg_paths
    except (EOFError,IOError) as e:
        with open(os.path.join(RESOURCE_DIR, file)) as fp:
            dataset = json.load(fp)
            # dataset = dataset[:10]
            questions = [i['uri']['question-id'] for i in dataset]
            questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
            pos_paths = []
            for i in dataset:
                path_id = i['parsed-data']['path_id']
                positive_path = []
                for p in path_id:
                    positive_path += [embeddings_interface.SPECIAL_CHARACTERS.index(p[0])]
                    positive_path += relations[int(p[1:])][3].tolist()
                pos_paths.append(positive_path)


            neg_paths = []
            for i in dataset:
                negative_paths_id = i['uri']['hop-2-properties'] + i['uri']['hop-1-properties']
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
                try:
                    negative_paths = np.random.choice(negative_paths,1000)
                except ValueError:
                    if len(negative_paths) == 0:
                        negative_paths = neg_paths[-1]
                        print "Using previous question's paths for this since no neg paths for this question."
                    index = np.random.randint(0, len(negative_paths), 1000)
                    negative_paths = np.array(negative_paths)
                    negative_paths = negative_paths[index]
                neg_paths.append(negative_paths)

            # neg_paths = [i[2] for i in dataset]
            #####################
            #Removing duplicates#
            #####################
            temp = neg_paths[0][0]
            for i in xrange(0, len(pos_paths)):
                to_remove = []
                for j in range(0,len(neg_paths[i])):
                    if np.array_equal(pos_paths[i], neg_paths[i][j]):
                        # to_remove.append(j)
                        if j != 0:
                            if not np.array_equal(pos_paths[i], neg_paths[i][j-1]):
                                neg_paths[i][j] = neg_paths[i][j-1]
                            else:
                                if j- 2 != 0:
                                    neg_paths[i][j] = neg_paths[i][j-2]
                                else:
                                    try:
                                        neg_paths[i][j] = neg_paths[i][j+1]
                                    except IndexError:
                                        neg_paths[i][j] = neg_paths[i][j-1]
                        else:
                            if not np.array_equal(pos_paths[i], neg_paths[i][j + 1]):
                                neg_paths[i][j] = neg_paths[i][j+1]
                            else:
                                try:
                                    neg_paths[i][j] = neg_paths[i][j+2]
                                except IndexError:
                                    neg_paths[i][j] = neg_paths[i][j+1]


                # neg_paths[i] = np.delete(neg_paths[i], to_remove) if to_remove else neg_paths[i]
            # for index in to_remove:


            neg_paths = [path for paths in neg_paths for path in paths]
            neg_paths = pad_sequences(neg_paths, maxlen=max_sequence_length, padding='post')
            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')

            all = np.concatenate([questions, pos_paths, neg_paths], axis=0)
            mapped_all, index = pd.factorize(all.flatten(), sort=True)
            mapped_all = mapped_all.reshape((-1, max_sequence_length))
            vectors = glove_embeddings[index]

            questions, pos_paths, neg_paths = np.split(mapped_all, [questions.shape[0], questions.shape[0]*2])
            neg_paths = np.reshape(neg_paths, (len(questions), NEGATIVE_SAMPLES, max_sequence_length))

            with open(os.path.join(RESOURCE_DIR, file + ".mapped.npz"), "w") as data, open(os.path.join(RESOURCE_DIR, file + ".index.npy"), "w") as idx:
                np.savez(data, questions, pos_paths, neg_paths)
                np.save(idx, index)

            return vectors, questions, pos_paths, neg_paths

def create_data_big_data(data,keras=True):
    '''
        The function takes id version of big data node and transforms it to version required by Keras network code.
    '''

    false_paths = []
    true_path = []
    id_paths = []
    for rel in data['parsed-data']['path_id']:
        true_path = true_path + [rel[0]] + reverse_relations[int(rel[1:])][-2]
        id_paths.append(data['parsed-data']['path_id'])
    for rel in data['uri']['hop-1-properties']:
        temp_path = [str(rel[0])] +  reverse_relations[int(rel[1])][-2]
        false_paths.append(temp_path)
        id_paths.append(temp_path)
    for rel in data['uri']['hop-2-properties']:
        temp_path = [str(rel[0])] +  reverse_relations[int(rel[1])][-2] + [str(rel[2])] +  reverse_relations[int(rel[3])][-2]
        false_paths.append(temp_path)
        id_paths.append(temp_path)
    question = data['parsed-data']['corrected_question']
    return [question,true_path,false_paths,id_paths]

def rel_id_to_rel(rel,relations):
    '''


    :param rel:
    :param relations: The relation lookup is inverse here
    :return:
    '''
    occurrences = []
    for key in relations:
        value = relations[key]
        if np.array_equal(value[3],np.asarray(rel)):
            occurrences.append(value)
    if len(occurrences) == 1:
        return occurrences[0][-1]
    else:
        '''
            prefers ontology over properties
        '''
        if 'property' in occurrences[0][3]:
            return occurrences[1][0]
        else:
            return occurrences[0][0]

def return_sign(sign):
    if sign == 2:
        return '+'
    else:
        return '-'

def id_to_path(path_id, vocab, relations, reverse_vocab, core_chain = True):
    '''


	:param path_id:  array([   3, 3106,    3,  647]) - corechain wihtout entity
	:param vocab: from continuous id space to discrete id space.
	:param relations: inverse relation lookup dictionary
	:return: paths
	'''

    # mapping from discrete space to continuous space.
    path_id = np.asarray([reverse_vocab[i] for i in path_id])
    #find all the relations in the given paths

    if core_chain:
        '''
            Identify the length. Is it one hop or two.
            The assumption is '+' is 2 and '-' is 3
        '''
        rel_length = 1
        if 2 in path_id[1:].tolist() or 3 in path_id[1:].tolist():
            rel_length = 2

        if rel_length == 2:
                sign_1 = path_id[0]
                try:
                    index_sign_2 = path_id[1:].tolist().index(2) + 1
                except ValueError:
                    index_sign_2 = path_id[1:].tolist().index(3) + 1
                rel_1,rel_2 = path_id[1:index_sign_2],path_id[index_sign_2+1:]
                rel_1 = rel_id_to_rel(rel_1,relations)
                rel_2 = rel_id_to_rel(rel_2,relations)
                sign_2 = index_sign_2
                path = [return_sign(sign_1),rel_1,return_sign(sign_2),rel_2]
                return path
        else:
            sign_1 = path_id[0]
            rel_1 = path_id[1:]
            rel_1 = rel_id_to_rel(rel_1,relations)
            path = [return_sign(sign_1),rel_1]
            return path
    else:
        variable = path_id[0]
        sign_1 = path_id[1]
        rel_1 = rel_id_to_rel(path_id[2:],relations)
        pass


def rdf_type_candidates(data,path_id, vocab, relations, reverse_vocab, only_x, core_chain = True):
    '''
        Takes in path ID (continous IDs, not glove vocab).
        And return type candidates (in continous IDs)
            based on whether we want URI or X candidates
    :param data:
    :param path_id:
    :param vocab:
    :param relations:
    :param reverse_vocab:
    :param core_chain:
    :return:
    '''

    #@TODO: Only generate specific candidates
    data = data['parsed-data']
    path = id_to_path(path_id, vocab, relations, reverse_vocab, core_chain = True)
    sparql = drt.reconstruct(data['entity'], path, alternative=True)
    sparqls = drt.create_sparql_constraints(sparql)

    if len(data['entity']) == 2:
        sparqls = [sparqls[1]]
    if len(path) == 2:
        sparqls = [sparqls[1]]
    type_x, type_uri = drt.retrive_answers(sparqls)

    # # Remove the "actual" constraint from this list (so that we only create negative samples)
    # try:
    #     type_x = [x for x in type_x if x not in data['constraints']['x']]
    # except KeyError:
    #     pass
    #
    # try:
    #     type_uri = [x for x in type_uri if x not in data['constraints']['uri']]
    # except KeyError:
    #     pass

    type_x_candidates, type_uri_candidates = drt.create_valid_paths(type_x, type_uri)
    # return type_x_candidates
    # Convert them to continous IDs.
    for i in range(len(type_x_candidates)):
        for j in range(len(type_x_candidates[i])):
            type_x_candidates[i][j] = vocab[type_x_candidates[i][j]]

    for i in range(len(type_uri_candidates)):
        for j in range(len(type_uri_candidates[i])):
            type_uri_candidates[i][j] = vocab[type_uri_candidates[i][j]]

    # Return based on given input.
    return type_x_candidates+type_uri_candidates
    # return type_x_candidates if only_x else type_uri_candidates


def load_relation(relation_file):
    relations = pickle.load(open(relation_file))
    inverse_relations = {}
    for key in relations:
        value = relations[key]
        new_key = value[0]
        value[0] = key
        value.append(new_key)
        inverse_relations[new_key] = value

    return inverse_relations


def create_true_positive_rdf_path(data):
    '''
        Creates true rdf-type constraint, if it exists.
        :param data: One specific node of id_data.
        :return: None if no true rdf-type constraint; else returns rdf-type constraint in continuous id space.
    '''

    datum = data
    if '?x' in datum['parsed-data']['constraints'].keys():
        pos_path = "x + " + dbp.get_label(datum['parsed-data']['constraints']['?x'])
    elif '?uri' in datum['parsed-data']['constraints'].keys():
        pos_path = "uri + " + dbp.get_label(datum['parsed-data']['constraints']['?uri'])
    else:
        return None
    pos_path = embeddings_interface.vocabularize(nlutils.tokenize(pos_path))
    for i in range(0,len(pos_path)):
        pos_path[i] = vocab[pos_path[i]]
    return pos_path

# Shuffle these matrices together @TODO this!
np.random.seed(0) # Random train/test splits stay the same between runs

# Divide the data into diff blocks
split_point = lambda x: int(len(x) * .80)

def train_split(x):
    return x[:split_point(x)]
def test_split(x):
    return x[split_point(x):]

with K.tf.device('/gpu:' + gpu):
    from keras.models import load_model
    metric = rank_precision_metric(10)
    model_corechain = load_model("./data/training/pairwise/model_32/model.h5", {'custom_loss':custom_loss, 'metric':metric})
    model_rdf_type_check = load_model("./data/training/rdf/model_00/model.h5", {'custom_loss':custom_loss, 'metric':metric})
    model_rdf_type_existence = load_model("./data/training/type-existence/model_00/model.h5")
    model_question_intent = load_model("./data/training/intent/model_00/model.h5")


def rdf_type_check(question,model_rdf_type_check, max_length = 30):
    '''

    :param question: vectorize question
    :param model_rdf_type_check: model
    :return:
    '''
    question_padded = np.zeros((1,max_length))
    try:
        question_padded[:,:question.shape[0]] = question
    except ValueError:
        question_padded = question[:,:question_padded.shape[0]]
    prediction = np.argmax(model_rdf_type_check.predict(question_padded))
    if prediction == 0:
        return True
    else:
        return False


def remove_positive_path(positive_path, negative_paths):
    new_negative_paths = []
    for i in range(0, len(negative_paths)):
        if not np.array_equal(negative_paths[i], positive_path):
            new_negative_paths.append(np.asarray(negative_paths[i]))
    return positive_path, np.asarray(new_negative_paths)


def load_reverse_rdf_type():
    rdf_type = json.load(open(LOC_RDF_TYPE_LOOKUP))
    rdf = {}
    for classes in rdf_type:
        rdf[classes] = embeddings_interface.vocabularize(nlutils.tokenize(dbp.get_label(classes)))
    return rdf


def convert_path_to_text(path):
    """
        Function used to convert a path (of continous IDs) to a text path.
        Eg. [ 5, 3, 420] : [uri, dbo:Poop]

    :param path: list of strings
    :return: list of text
    """

    # First we need to convert path to glove vocab
    path = [reverse_vocab[x] for x in path]

    # Then to convert this to text
    var = ''
    for key in embeddings_interface.glove_vocab.keys():
        if embeddings_interface.glove_vocab[key] == path[0]:
            var = key
            break

    dbo_class = ''
    for key in reverse_rdf_dict.keys():
        if list(reverse_rdf_dict[key]) == list(path[2:]):
            dbo_class = key
            break

    return [var, dbo_class]



def construct_paths(data):
    '''
    :param data: a data node of id_big_data
    :return: unpadded , continous id spaced question, positive path, negative paths
    '''
    question = np.asarray(data['uri']['question-id'])
    # questions = pad_sequences([question], maxlen=max_length, padding='post')

    # inverse id version of positive path and creating a numpy version
    positive_path_id = data['parsed-data']['path_id']
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
    positive_path, negative_paths = remove_positive_path(positive_path, negative_paths)

    # remap all the id's to the continous id space.

    # passing all the elements through vocab
    question = np.asarray([vocab[key] for key in question])
    positive_path = np.asarray([vocab[key] for key in positive_path])
    for i in range(0, len(negative_paths)):
        # temp = []
        for j in xrange(0, len(negative_paths[i])):
            try:
                negative_paths[i][j] = vocab[negative_paths[i][j]]
            except:
                negative_paths[i][j] = vocab[0]
                # negative_paths[i] = np.asarray(temp)
                # negative_paths[i] = np.asarray([vocab[key] for key in negative_paths[i] if key in vocab.keys()])

    return question,positive_path,negative_paths




def question_intent(padded_question):
    '''
        predicting the intent of the question.
            List, Ask, Count.
    '''
    intent = np.argmax(model_question_intent.predict(padded_question))
    return ['ask', 'count', 'list'][intent]

def rdf_constraint_check(padded_question):
    '''
        predicting the existence of rdf type constraints.
    '''
    return np.argmax(model_rdf_type_existence.predict(padded_question))


def padded_question(question):
    '''

    :param question: continuous space id question
    :return: padded question
    '''
    padded_question = np.zeros(max_length)
    padded_question[:min(max_length, len(question))] = question[:min(max_length, len(question))]
    padded_question = padded_question.reshape((1, padded_question.shape[0]))
    return padded_question

id_data = json.load(open('resources_v8/id_big_data.json'))
vocab = pickle.load(open('resources_v8/id_big_data.json.vocab.pickle'))
relations = load_relation('resources_v8/relations.pickle')
reverse_rdf_dict = load_reverse_rdf_type()

reverse_vocab = {}
for keys in vocab:
    reverse_vocab[vocab[keys]] = keys

id_data_test = test_split(id_data)
id_data_train = train_split(id_data)

core_chain_correct = []
max_length = 25

progbar = ProgressBar()
iterator = progbar(id_data_train)


print "the length of train data is ", len(id_data_test)
core_chain_counter = 0
for data in iterator:

    #some macros which would be useful for constructing sparqls
    sparql_reconstruction = {}
    rdf_type = True

    question, positive_path, negative_paths = construct_paths(data)


    if len(negative_paths) == 0:
        best_path = positive_path
    else:
        '''
            The output is made by stacking positive path over negative paths.
        '''
        output = rank_precision_runtime(model_corechain, question, positive_path,
                                        negative_paths, 10000, max_length)
        if np.argmax(output) == 0:
            core_chain_counter = core_chain_counter +  1
            print core_chain_counter

        best_path_index = np.argmax(output)
        if best_path_index == 0:
            best_path = positive_path
        else:
            best_path = negative_paths[best_path_index-1]

    #best path is in continuous id space.
    sparql_reconstruction['best_path'] = best_path

    padded_question = padded_question(question)

    #Predecting the intent
    intent = question_intent(padded_question)
    if DEBUG: print "Intent of the question is : ", str(intent)
    sparql_reconstruction['intent'] = intent

    # Predecting the rdf-constraint
    rdf_constraint = rdf_constraint_check(padded_question)
    if DEBUG: print "Rdf constraint of the question is : ", str(rdf_constraint)
    sparql_reconstruction['rdf_constraint'] = False if rdf_constraint == 2 else sparql_reconstruction['rdf_constraint'] = True

    sparql_reconstruction['rdf_constraint_type'] = ['x','uri','none'][rdf_constraint]

    if rdf_constraint != 2:

        rdf_candidates = rdf_type_candidates(data, best_path, vocab, relations, reverse_vocab, only_x=rdf_constraint == 0,
                                             core_chain=True)
        '''
			Predicting the rdf type constraints for the best core chain.
		'''
        if rdf_candidates:
            output = rank_precision_runtime(model_rdf_type_check, question, rdf_candidates[0],
                                            rdf_candidates, 180, max_length)
            # rdf_best_path = convert_path_to_text(rdf_candidates[np.argmax(output[1:])])
            sparql_reconstruction['rdf_best_path'] = rdf_candidates[np.argmax(output[1:])+1]
        else:
            sparql_reconstruction['rdf_best_path'] = []

    print sparql_reconstruction