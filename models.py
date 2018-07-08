import os
import numpy as np
import sys
from abc import abstractmethod
import torch
from torch import nn
import pickle
import math
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

DEBUG = True
DATA_DIR = './data/training/pairwise'
RESOURCE_DIR = './resources'
EPOCHS = 300
BATCH_SIZE = 880 # Around 11 splits for full training dataset
LEARNING_RATE = 0.001
LOSS = 'categorical_crossentropy'
NEGATIVE_SAMPLES = 1000


class TunableEmbedding(nn.Module):
    def cudafy(self, x):
        return x.cuda(self.gpu) if self.gpu is not None else x


    def __init__(self, vectors, dim_in, dim_out, dim_tune=5000, dropout=0.0, gpu=0):
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.embed = nn.Embedding(
            dim_in,
            dim_out
        )
        self.embed = nn.Dropout(dropout)(self.embed)

        self.tune = nn.Embedding(
            dim_tune,
            dim_out
        )
        self.tune = nn.Dropout(dropout)(self.tune)
        self.mod_ids = lambda sent: sent % (dim_tune-1)+1

        self.embeddings = [self.embed, self.tune]
        self.initialize_embeddings()

        if gpu is not None:
            self.cuda(gpu)


    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm_(p=2, dim=0, maxnorm=1)



    def initialize_embeddings(self):
        r = 6/np.sqrt(self.dim_out)

        for e in self.embeddings:
            e.weight.data.uniform_(-r, r)

        self.normalize_embeddings()

    def forward(self, sentence):
        mod_sent = self.mod_ids(sentence)
        tuning = self.tune(mod_sent)
        pretrained = self.embed(sentence)
        vectors = torch.sum(tuning, pretrained)
        return vectors

class ScoreModel(nn.Network):
    def __init__(self, vocabulary_size, hidden_size, gpu=0, dropout=0.5):
        self.gpu = gpu
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.embed = TunableEmbedding(vocabulary_size, hidden_size, dropout=dropout)
        self.encode = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, dropout=dropout)

        if gpu is not None:
            self.cuda(gpu)

    def encode(self, x):
        embeddings = self.embed(x)
        _, (h, _) = self.encode(embeddings)
        h = h.resize(-1, 2*self.hidden_size)
        return h

    def forward(self, question, path):
        question = self.encode(question)
        path = self.encode(path)
        return torch.sum(question*path, -1)

class PairwiseRankingModel(nn.Network):
    def cudafy(self, x):
        return x.cuda(self.gpu) if self.gpu is not None else x

    def __init__(self, vocabulary_size, hidden_size, gpu=0, dropout=0.5):
        self.gpu = gpu
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.score = ScoreModel(vocabulary_size, hidden_size, gpu, dropout)

        if gpu is not None:
            self.cuda(gpu)

    def forward(self, question, pos_path, neg_path):
        pos_score = self.score(question, pos_path)
        neg_score = self.score(question, neg_path)
        zero = self.cudafy(torch.FloatTensor([0.0]))
        criterion = torch.mean(torch.max(Variable(zero), 1.0 - pos_score + neg_score))
        return criterion



def get_glove_embeddings():
    from utils.embeddings_interface import __check_prepared__
    __check_prepared__('glove')

    from utils.embeddings_interface import glove_embeddings
    return glove_embeddings


def load_data(file, max_sequence_length):
    glove_embeddings = get_glove_embeddings()

    try:
        with open(os.path.join(RESOURCE_DIR, file + ".mapped.npz")) as data, open(os.path.join(RESOURCE_DIR, file + ".index.npy")) as idx:
            dataset = np.load(data)
            questions, pos_paths, neg_paths = dataset['arr_0'], dataset['arr_1'], dataset['arr_2']
            index = np.load(idx)
            vectors = glove_embeddings[index]
            return vectors, questions, pos_paths, neg_paths
    except:
        with open(os.path.join(RESOURCE_DIR, file)) as fp:
            dataset = pickle.load(fp)
            questions = [i[0] for i in dataset]
            questions = pad_sequences(questions, maxlen=max_sequence_length, padding='post')
            pos_paths = [i[1] for i in dataset]
            pos_paths = pad_sequences(pos_paths, maxlen=max_sequence_length, padding='post')
            neg_paths = [i[2] for i in dataset]
            neg_paths = [path for paths in neg_paths for path in paths]
            neg_paths = pad_sequences(neg_paths, maxlen=max_sequence_length, padding='post')

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


class TrainingDataset(Dataset):
    def __init__(self, questions, pos_paths, neg_paths, max_length, neg_paths_per_epoch, batch_size):
        self.dummy_y = np.zeros(batch_size)
        self.firstDone = False
        self.max_length = max_length
        self.neg_paths_per_epoch = neg_paths_per_epoch

        self.questions = np.reshape(np.repeat(np.reshape(questions,
                                            (questions.shape[0], 1, questions.shape[1])),
                                 neg_paths_per_epoch, axis=1), (-1, max_length))

        self.pos_paths = np.reshape(np.repeat(np.reshape(pos_paths,
                                            (pos_paths.shape[0], 1, pos_paths.shape[1])),
                                 neg_paths_per_epoch, axis=1), (-1, max_length))

        self.neg_paths = neg_paths

        self.neg_paths_sampled = np.reshape(self.neg_paths[:,np.random.randint(0, NEGATIVE_SAMPLES, self.neg_paths_per_epoch), :],
                                            (-1, self.max_length))

        self.questions_shuffled, self.pos_paths_shuffled, self.neg_paths_shuffled = \
            shuffle(self.questions, self.pos_paths, self.neg_paths_sampled)

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.questions) / self.batch_size)

    def __getitem__(self, idx):
        index = lambda x: torch.from_numpy(x[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_questions = index(self.questions_shuffled)
        batch_pos_paths = index(self.pos_paths_shuffled)
        batch_neg_paths = index(self.neg_paths_shuffled)

        return ([batch_questions, batch_pos_paths, batch_neg_paths], self.dummy_y)

class ValidationDataset(Dataset):
    def __init__(self, questions, pos_paths, neg_paths, max_length, neg_paths_per_epoch, batch_size):
        self.dummy_y = np.zeros(batch_size)
        self.firstDone = False
        self.max_length = max_length
        self.neg_paths_per_epoch = neg_paths_per_epoch

        self.questions = np.reshape(np.repeat(np.reshape(questions,
                                            (questions.shape[0], 1, questions.shape[1])),
                                 neg_paths_per_epoch+1, axis=1), (-1, max_length))

        self.pos_paths = np.reshape(pos_paths,
                                            (pos_paths.shape[0], 1, pos_paths.shape[1]))
        self.neg_paths = neg_paths
        neg_paths_sampled = self.neg_paths[:, np.random.randint(0, NEGATIVE_SAMPLES, self.neg_paths_per_epoch), :]
        self.all_paths = np.reshape(np.concatenate([self.pos_paths, neg_paths_sampled], axis=1), (-1, self.max_length))

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.questions) / self.batch_size)

    def __getitem__(self, idx):
        index = lambda x: x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_questions = index(self.questions)
        batch_all_paths = index(self.all_paths)

        return ([batch_questions, batch_all_paths, np.zeros_like(batch_all_paths)], self.dummy_y)


def main():
    gpu = sys.argv[1]

    """
        Data Time!
    """
    # Pull the data up from disk
    max_length = 50

    vectors, questions, pos_paths, neg_paths = load_data("id_results.pickle", max_length)

    np.random.seed(0) # Random train/test splits stay the same between runs

    # Divide the data into diff blocks
    split_point = lambda x: int(len(x) * .80)

    def train_split(x):
        return x[:split_point(x)]
    def test_split(x):
        return x[split_point(x):]

    train_pos_paths = train_split(pos_paths)
    train_neg_paths = train_split(neg_paths)
    train_questions = train_split(questions)

    test_pos_paths = test_split(pos_paths)
    test_neg_paths = test_split(neg_paths)
    test_questions = test_split(questions)

    neg_paths_per_epoch_train = 10
    neg_paths_per_epoch_test = 1000
    dummy_y_train = np.zeros(len(train_questions)*neg_paths_per_epoch_train)
    dummy_y_test = np.zeros(len(test_questions)*(neg_paths_per_epoch_test+1))

    print train_questions.shape
    print train_pos_paths.shape
    print train_neg_paths.shape

    print test_questions.shape
    print test_pos_paths.shape
    print test_neg_paths.shape

    neg_paths_per_epoch_train = 10
    neg_paths_per_epoch_test = 1000

    trainLoader = DataLoader(TrainingDataset(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_train, BATCH_SIZE))
    validLoader = DataLoader(ValidationDataset(train_questions, train_pos_paths, train_neg_paths,
                                                  max_length, neg_paths_per_epoch_test, 9999))



    model = PairwiseRankingModel()
    optimizer = optim.Adam(model.parameters())




class QARankingModel:
    '''
    Abstract class for pairwise ranking based question answering models
    '''
    def __init__(self, max_sequence_length, data_dir, similarity='dot', dropout=0.5):
        self._data_dir = data_dir

        self.question = Input(shape=(max_sequence_length,), dtype='int32', name='question_input')
        self.pos_path = Input(shape=(max_sequence_length,), dtype='int32', name='pos_path_input')
        self.neg_path = Input(shape=(max_sequence_length,), dtype='int32', name='neg_path_input')
        self._path = Input(shape=(max_sequence_length,), dtype='int32', name='path_stub')

        self.dropout = dropout
        self.similarity = similarity
        self.max_sequence_length = max_sequence_length

        # initialize a bunch of variables that will be set later
        self._question_score = None
        self._path_score = None
        self._similarities = None
        self._score_model = None

        self.training_model = None
        self.prediction_model = None

    @abstractmethod
    def build(self):
        return

    def get_similarity(self):
        '''
        Specify similarity in configuration under 'similarity' -> 'mode'
        If a parameter is needed for the model, specify it in 'similarity'
        Example configuration:
        config = {
            ... other parameters ...
            'similarity': {
                'mode': 'gesd',
                'gamma': 1,
                'c': 1,
            }
        }
        cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
        polynomial: (gamma * dot(a, b) + c) ^ d
        sigmoid: tanh(gamma * dot(a, b) + c)
        rbf: exp(-gamma * l2_norm(a-b) ^ 2)
        euclidean: 1 / (1 + l2_norm(a - b))
        exponential: exp(-gamma * l2_norm(a - b))
        gesd: euclidean * sigmoid
        aesd: (euclidean + sigmoid) / 2
        '''

        params = self.params
        similarity = params['mode']

        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        l2_norm = lambda a, b: K.sqrt(K.sum(K.square(a - b), axis=1, keepdims=True))

        if similarity == 'dot':
            return lambda x: dot(x[0], x[1])
        elif similarity == 'cosine':
            return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())
        elif similarity == 'polynomial':
            return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
        elif similarity == 'sigmoid':
            return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
        elif similarity == 'rbf':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
        elif similarity == 'euclidean':
            return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        elif similarity == 'exponential':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
        elif similarity == 'gesd':
            euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) * sigmoid(x)
        elif similarity == 'aesd':
            euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) + sigmoid(x)
        else:
            raise Exception('Invalid similarity: {}'.format(similarity))

    def get_score_model(self):
        if None in [self._question_score, self._path_score]:
            self._question_score, self._path_score = self.build()

        if self._score_model is None:
            dropout = Dropout(self.dropout)
            similarity = self.get_similarity()

            qa_model = Lambda(similarity, output_shape=lambda _: (None, 1))([dropout(self._question_score),
                                                                             dropout(self._path_score)])
            self._score_model = Model(inputs=[self.question, self._path], outputs=qa_model, name='qa_model')

        return self._score_model

    def get_training_model(self):
        if not self.training_model:
            score_model = self.get_score_model()
            pos_score = score_model([self.question, self.pos_path])
            neg_score = score_model([self.question, self.neg_path])
            loss = Lambda(lambda x: K.relu(self.config['margin'] - x[0] + x[1]),
                      output_shape=lambda x: x[0])([pos_score, neg_score])
            self.training_model = Model(inputs=[self.question, self.pos_path, self.neg_path], outputs=loss,
                                    name='training_model')
        return self.training_model

    def get_prediction_model(self):
        if not self.prediction_model:
            score_model = self.get_score_model()
            score = score_model([self.question, self.pos_path])
            self.prediction_model = Model(inputs=[self.question, self.pos_path], outputs=score,
                                          name='prediction_model')
        return self.prediction_model


    def compile(self, optimizer, **kwargs):
        self.prediction_model.compile(loss=lambda y_true, y_pred: y_pred,
                                      optimizer=optimizer, **kwargs)
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred,
                                    metrics=[self.get_hitsatk_metric(1),
                                             self.get_hitsatk_metric(5),
                                             self.get_hitsatk_metric(10)],
                                    optimizer=optimizer, **kwargs)

    def fit(self, training_data, validation_data, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        assert isinstance(self.prediction_model, Model)
        return self.training_model.fit_generator(training_data, validation_data=validation_data, **kwargs)

    def predict(self, x, **kwargs):
        assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
        return self.prediction_model.predict(x, **kwargs)

    def get_next_save_path(self, **kwargs):
        # Find the current model dirs in the data dir.
        _, dirs, _ = os.walk(self._data_dir).next()

        # If no folder found in there, create a new one.
        if len(dirs) == 0:
            os.mkdir(os.path.join(self._data_dir, "model_00"))
            dirs = ["model_00"]

        # Find the latest folder in here
        dir_nums = sorted([ x[-2:] for x in dirs])
        l_dir = os.path.join(self._data_dir, "model_" + dir_nums[-1])

        # Create new folder with name model_(i+1)

        new_num = int(dir_nums[-1]) + 1
        if new_num < 10:
            new_num = str('0') + str(new_num)
        else:
            new_num = str(new_num)

        l_dir = os.path.join(self._data_dir, "model_" + new_num)
        os.mkdir(l_dir)

        return l_dir

    def save_model(self, path):
        self.prediction_model.save(path)

    def load_model(self, path):
        self.prediction_model = keras.models.load_model(path)

    def get_hitsatk_metric(self, k):
        neg_samples = self.neg_samples
        def metric(y_true, y_pred):
            scores = y_pred[:, 0]
            scores = K.reshape(scores, (-1, neg_samples+1))
            _, topk = K.tf.nn.top_k(scores, k=10, sorted=True)
            hitsatk = K.cast(K.shape(K.tf.where(K.tf.equal(topk,0)))[0], 'float32')
            hitsatk = hitsatk/K.cast(K.shape(scores)[0], 'float32')
            return hitsatk

        # dirty exec hack to create function named hits_at_k because keras extracts function name
        exec "def hits_at_%d(y_true, y_pred): return metric(y_true, y_pred)" % k in locals()
        return locals()["hits_at_%d" % k]
