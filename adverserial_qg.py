from spacy.tokenizer import Tokenizer
from nltk.corpus import wordnet  # Import wordnet from the NLTK
import numpy as np
import spacy, json



nlp = spacy.load("en_core_web_sm")
question_intent_list = ['who', 'how', 'count','where', 'what', 'tell']
tokenizer = Tokenizer(nlp.vocab)


def get_relchunks(sentence, nlp):
    doc =  nlp(sentence)
    # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    # print("entites are:", [entity.text for entity in doc.ents])

    final_rel_list = []

    entity_text = [entity.text for entity in doc.ents]
    if len(entity_text) > 1:
        flat_entity_text = [word for e in entity_text for word in e]
    else:
        flat_entity_text = entity_text
    for chunk in doc.noun_chunks:
        found_flag = False
        for c in chunk.text.split(" "):
            if c in flat_entity_text:
                found_flag = True
            if c.lower() in question_intent_list:
                found_flag = True
        if not found_flag:
            final_rel_list.append(chunk.text)

    return final_rel_list



def get_syn(word):

    syn = list()
    ant = list()
    for synset in wordnet.synsets(word.lower()):
        for lemma in synset.lemmas():
            syn.append(lemma.name())  # add the synonyms

    return syn



def find_syn(word,vocab):
    syn = get_syn(word.lower())
    for s in syn:
        s = s.lower()
        s_tokens = s.split("_")
        found = []
        for i in s_tokens:
            if i.lower() not in vocab:
                found.append(i)

        if s_tokens != [word.lower()]:
            if not found:
                print(word.lower(),s_tokens)
                return s_tokens
    return []


def load_dataset():
    data = json.load(open('resources/lcquad_data_set.json'))
    return data[int(len(data)*.80):]

def load_vocab():
    vocab = np.load('resources/vocab.npy.npz')['arr_0']
    return [word for word in vocab]

test_data = load_dataset()
vocab = load_vocab()

def random_drop(sentence, nlp, tokenizer):
    doc = nlp(sentence)
    tokens = tokenizer(sentence)
    drop_index = np.random.randint(0,len(tokens)-1)
    new_tokens = []
    for index, tok in enumerate(tokens):
        if index != drop_index:
            new_tokens.append(tok.text)

    return " ".join(new_tokens).lower()

def synonym_replacement(sentence, nlp, tokenizer):

    tokens = tokenizer(sentence)
    max_count = 6
    while True:
        replace_index = np.random.randint(0, len(tokens) - 1)
        syns = get_syn(tokens[replace_index].text)

        if syns:
            if len(syns) == 1:
                syn_word = syns[0]
            else:
                syn_word = syns[np.random.randint(0,len(syns)-1)]
            syn_word = syn_word.split("_")
            new_tokens = []
            for index, tok in enumerate(tokens):
                if index != replace_index:
                    new_tokens.append(tok.text)
                else:
                    if len(syn_word) > 1:
                        for s in syn_word:
                            new_tokens.append(s)
                    else:
                        new_tokens.append(syn_word[0])
            new_tokens = [n.lower() for n in new_tokens]
            return (" ".join(new_tokens),True)
        else:
            if max_count == 0:
                return (sentence,False)
            else:
                max_count = max_count - 1


# def find_and_check_syn(word,vocab):


def synonym_replacement_vocab(sentence, nlp, tokenizer,vocab):
    tokens = tokenizer(sentence)
    max_count = 6
    while True:



        counter = 5

        while counter > 0:
            replace_index = np.random.randint(1, len(tokens) - 1)
            syns = find_syn(tokens[replace_index].text,vocab)

            if syns:
                counter = 0
            else:
                counter = counter - 1

        print(syns)
        if syns:
            # syn_word = syns[np.random.randint(0,len(syns)-1)]
            # syn_word = syn_word.split("_")
            # syn_word = [s.lower() for s in syn_word]
            #
            # not_found = True
            #
            #
            # for s in syn_word:
            #     if s not in vocab:
            #         found = False

            syn_word = syns
            new_tokens = []
            for index, tok in enumerate(tokens):
                if index != replace_index:
                    new_tokens.append(tok.text)
                else:
                    if len(syn_word) > 1:
                        for s in syn_word:
                            new_tokens.append(s)
                    else:
                        new_tokens.append(syn_word[0])
            return (" ".join(new_tokens),True)


        else:
            if max_count == 0:
                return (sentence,False)
            else:
                max_count = max_count - 1



syn_counter = 0
syn_vocab_counter = 0

for index, d in enumerate(test_data):
    syn_replacement, c = synonym_replacement(d['corrected_question'], nlp, tokenizer)
    syn_replacement_vocab, v = synonym_replacement_vocab(d['corrected_question'], nlp, tokenizer,vocab)
    syn_counter = c + syn_counter
    syn_vocab_counter = v + syn_vocab_counter
    dropped = random_drop(d['corrected_question'],nlp,tokenizer)
    d['syn_replacement'] = syn_replacement
    d['syn_replacement_vocab'] = syn_replacement_vocab
    d['dropped'] = dropped
    test_data[index] = d