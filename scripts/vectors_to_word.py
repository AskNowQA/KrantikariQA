'''

    Loads vector file, and creates another file word , which corresponds to specific word.
'''
import pickle
import numpy as np

VECTOR_FILE = '../data/data/common/vectors.npy'
VOCAB_FILE = '../data/data/common/vocab.pickle'
GLOVE_VOCAB_FILE = '../resources/glove_vocab.pickle'

id_to_gloveid = {}
gloveid_to_word = {}
id_to_embedding = np.load(VECTOR_FILE)


word_to_gloveid = pickle.load(open(GLOVE_VOCAB_FILE))
gloveid_to_id =  pickle.load(open(VOCAB_FILE))


#reversing for our purrrpose

for keys in word_to_gloveid:
    gloveid_to_word[word_to_gloveid[keys]] = keys

for keys in gloveid_to_id:
    id_to_gloveid[gloveid_to_id[keys]] = keys

word_list = []

for id in range(len(id_to_embedding)):
    gloveid = id_to_gloveid[id]
    word = gloveid_to_word[gloveid]
    word_list.append(word)

pickle.dump(word_list,open('../data/data/common/glove.300d.words','w+'))

#rsync -avz --progress scripts/vectors_to_word.py qrowdgpu+titan:/shared/home/GauravMaheshwari/new_kranti/KrantikariQA/scripts/