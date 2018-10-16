'''
    In glove vocab glove_vocab = pickle.load(open('resources/glove_vocab.pickle'))
        Update glove_vocab[0] = pad to  glove_vocab[0] = <MASK>

    After this run vectors_to_word.py file.

    This hack is to include migrate present code to quelos code
'''

import pickle
GLOVE_VOCAB = 'resources/glove_vocab.pickle'

#Updating the glove vocab file
glove_vocab = pickle.load(open(GLOVE_VOCAB))
glove_vocab.pop('PAD')
glove_vocab['<MASK>'] = 0
pickle.dump(glove_vocab,open(GLOVE_VOCAB,'w+'))



