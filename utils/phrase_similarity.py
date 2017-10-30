'''
    calculates the simialrity between two phrases using word2vec

'''
#TODO: Check how well is this performing

import gensim
import numpy as np
import traceback

model = gensim.models.KeyedVectors.load_word2vec_format('/home/gaurav/Downloads/models/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

def ConvertVectorSetToVecAverageBased(vectorSet, ignore = []):
		if len(ignore) == 0:
			return np.mean(vectorSet, axis = 0)
		else:
			return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)


def phrase_similarity(_phrase_1, _phrase_2):
    phrase_1 = _phrase_1.split(" ")
    phrase_2 = _phrase_2.split(" ")
    vw_phrase_1 = []
    vw_phrase_2 = []
    for phrase in phrase_1:
        try:
            # print phrase
            vw_phrase_1.append(model.word_vec(phrase.lower()))
        except:
            # print traceback.print_exc()
            continue
    for phrase in phrase_2:
        try:
            vw_phrase_2.append(model.word_vec(phrase.lower()))
        except:
            continue
    if len(vw_phrase_1) == 0 or len(vw_phrase_2) == 0:
        return 0
    v_phrase_1 = ConvertVectorSetToVecAverageBased(vw_phrase_1)
    v_phrase_2 = ConvertVectorSetToVecAverageBased(vw_phrase_2)
    cosine_similarity = np.dot(v_phrase_1, v_phrase_2) / (np.linalg.norm(v_phrase_1) * np.linalg.norm(v_phrase_2))
    return float(cosine_similarity)

print type(float(phrase_similarity("military branchere","child organization")))