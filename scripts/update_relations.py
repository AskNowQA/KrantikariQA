import pickle
from utils import embeddings_interface


relation_file = 'data/data/common/relations.pickle'
relations = pickle.load(open(relation_file))


'''
    The file loaded has the structure
    
    
    relations['http://dbpedia.org/property/juniorHighPrincipal'] = [0,'junior High Principal',['junior', 'High', 'Principal'],array([2683,  194, 3930])]

'''

vocabulary = []


for rel in relations.keys():
    tokenized = relations[rel][2]
    vocabulary = tokenized + vocabulary
    vocabulary = list(set(vocabulary))

vocabulary = [v.lower() for v in vocabulary]
vocabulary = list(set(vocabulary))


'''
    Updates the vocabulary 
'''
print("updating vocab")
embeddings_interface.update_vocabulary(vocabulary)
print("saving files")
embeddings_interface.save()
print("saved")

for rel in relations.keys():
    tokenized = relations[rel][2]
    relations[rel][3] = embeddings_interface.vocabularize(tokenized)

pickle.dump(relations,open(relation_file,'w+'))

