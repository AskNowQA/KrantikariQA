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
rel_keys = [k.lower() for k in relations.keys()]
vocabulary = list(set(vocabulary))
vocabulary = vocabulary + rel_keys


'''
    Updates the vocabulary 
'''
print("updating vocab")
print("len voab is", len(vocabulary))
embeddings_interface.update_vocabulary(vocabulary)
print("saving files")
embeddings_interface.save()
print("saved")

for rel in relations.keys():
    tokenized = relations[rel][2]
    relations[rel][3] = embeddings_interface.vocabularize(tokenized)
    try:
        relations[rel][4] = embeddings_interface.vocabularize([rel.lower()])
    except:
        relations[rel].append(embeddings_interface.vocabularize([rel.lower()]))

pickle.dump(relations,open(relation_file,'w+'))
