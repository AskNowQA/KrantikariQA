'''
    Note that this script needs to be run via ipython from root directory.
    Since the question id in qald is out of syn. The file takes
        > data/data/qald/qald_id_big_data_train.json
        > data/data/qald/qald_id_big_data_test.json
    and updates the
        data[0]['uri']['question-id']
'''
import json,pickle
from utils import natural_language_utilities as nlutils

TRAIN_QALD_FILE = 'data/data/qald/qald_id_big_data_train.json'
TEST_QALD_FILE = 'data/data/qald/qald_id_big_data_test.json'
GLOVE_VOCAB_FILE = 'resources/glove_vocab.pickle'



train_file = json.load(open(TRAIN_QALD_FILE))
test_file = json.load(open(TEST_QALD_FILE))
word_to_gloveid = pickle.load(open(GLOVE_VOCAB_FILE))


def update_id(data,word_to_gloveid):
    for index in range(len(data)):
        question = data[index]['parsed-data']['corrected_question']
        updated_question_id = []
        for word in nlutils.tokenize(question):
            try:
                updated_question_id.append(word_to_gloveid[word.lower()])
            except:
                updated_question_id.append(1)
        data[index]['uri']['question-id'] = updated_question_id
    return data

print("updateing question id for train file ")
data = update_id(train_file,word_to_gloveid)
print("dumping the train data")
json.dump(data,open(TRAIN_QALD_FILE,'w+'))
print("updating question id for test file")
data = update_id(test_file,word_to_gloveid)
print("dumping the train data")
json.dump(data,open(TEST_QALD_FILE,'w+'))