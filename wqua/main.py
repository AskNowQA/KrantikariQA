'''

This files attempts to create question answering just using word embedding.
'''
import json
import os
from pprint import pprint

path =  os.path.basename(os.getcwd())
os.chdir(  "../" + path  )
# sys.path.insert(0, '../')

import utils.dbpedia_interface as db_interface
import utils.natural_language_utilities as nlutils
dbp = db_interface.DBPedia(_verbose=True, caching=False)

import utils.phrase_similarity as sim


DEBUG = True
black_list = ['http://www.w3.org/2000/01/rdf-schema#seeAlso','http://purl.org/linguistics/gold/hypernym',
              'http://www.w3.org/1999/02/22-rdf-syntax-ns#type','http://www.w3.org/2000/01/rdf-schema#label',
              'http://www.w3.org/2000/01/rdf-schema#comment','http://purl.org/voc/vrank#hasRank','http://xmlns.com/foaf/0.1/isPrimaryTopicOf',
              'http://xmlns.com/foaf/0.1/primaryTopic']


def vectorize(_tokens, _report_unks = False):
    """
        Function to embed a sentence and return it as a list of vectors.
        WARNING: Give it already split. I ain't splitting it for ye.

        :param _input: The sentence you want embedded. (Assumed pre-tokenized input)
        :param _report_unks: Whether or not return the out of vocab words
        :return: Numpy tensor of n * 300d, [OPTIONAL] List(str) of tokens out of vocabulary.
    """

    # # Cleaned sentence
    # cleaned_input = _input.replace("?", "").replace(",", "").strip()
    #
    # # Split the sentence into word tokens
    # # @TODO: Use a proper tokenizer.
    # tokens = cleaned_input.split()

    # Logic for Glove
    op = []
    unks = []
    for token in _tokens:

        # Small cap everything
        token = token.lower()

        if token == "+":
            token_embedding = np.repeat(1,300)
        elif token == "-":
            token_embedding = np.repeat(-1, 300)
        else:
            try:
                if EMBEDDING == "GLOVE":
                    token_embedding = embedding_glove[token]
                elif EMBEDDING == 'WORD2VEC':
                    token_embedding = embedding_word2vec[token]

            except KeyError:
                if _report_unks:
                    unks.append(token)
                token_embedding = np.zeros(300, dtype=np.float32)

        op += [token_embedding]

    # if DEBUG: print _tokens, "\n",

    return (np.asarray(op), unks) if _report_unks else np.asarray(op)


def tokenize(_input, _ignore_brackets = False):
    """
        Tokenize a question.
        Changes:
            - removes question marks
            - removes commas
            - removes trailing spaces
            - can remove text inside one-level brackets.

        @TODO: Improve tokenization
        :param _input: str, _ignore_brackets: bool
        :return: list of tokens
    """
    cleaner_input = _input.replace("?", "").replace(",", "").strip()
    if _ignore_brackets:
        # If there's some text b/w brackets, remove it. @TODO: NESTED parenthesis not covered.
        pattern = r'\([^\)]*\)'
        matcher = re.search(pattern, cleaner_input, 0)

        if matcher:
            substring = matcher.group()

            cleaner_input = cleaner_input[:cleaner_input.index(substring)] + cleaner_input[cleaner_input.index(substring) + len(substring):]

    return cleaner_input.strip().split()


def question_vector(question):
    '''
		:param question: String of question
		:return: vector of question.
	'''
    return vectorize(tokenize(question))

def parse(file_dir,debug=False):
    if debug:
        print "the file dir is ", file_dir

def top_k_relation(entity,question,relations,k=5,method=1):
    '''
    :param entity: The entity url in the question
    :param question: The vector form of the question
    :param relations: A list of tuple with the first being the whole relation url and the secodn being incoming or outgoing relation
    :param k: the top k choices
    :return: a list of tuple of relations
    '''
    entity_label = dbp.get_label(entity)
    temp_relations = []
    for rel_tup in relations:
        if rel_tup[0] not in black_list:
            #Find the similarity between the ent+rel and the question
            if method == 1:
                phrase_1 = entity_label + " " + nlutils.get_label_via_parsing(rel_tup[0])
                similarity_score = sim.phrase_similarity(phrase_1, question)
                temp_relations.append((rel_tup[0],rel_tup[1],similarity_score))
            if method == 2:
                phrase_1 = nlutils.get_label_via_parsing(rel_tup[0])
                similarity_score = sim.phrase_similarity(phrase_1, question)
                temp_relations.append((rel_tup[0], rel_tup[1], similarity_score))
        else:
            continue
    temp_relations = sorted(temp_relations, key=lambda tup: tup[2],reverse=True)
    if len(temp_relations) > k:
        return temp_relations[:k]
    else:
        return temp_relations


def updated_get_relationship_hop(_entity, _relation):
    '''

        This function gives all the relations after the _relationship chain. The
        difference in this and the get_relationship_hop is that it gives all the relationships from _relations[:-1],
    '''
    entities = [_entity]    #entites are getting pushed here
    for rel in _relation:
        outgoing = rel[1]
        if outgoing == "outgoing":
            ''' get the objects '''
            temp = [dbp.get_entity(ent,rel,outgoing=True) for ent in entities]
            entities = list(set([item for sublist in temp for item in sublist]))
        else:
            '''get the subjects '''
            temp = [dbp.get_entity(ent, rel, outgoing=False) for ent in entities]
            entities = list(set([item for sublist in temp for item in sublist]))
        temp_ent = []
        ''' If after the query we get a literal instead of a resources'''
        for ent in entities:
            if "http://dbpedia.org/resource" in ent:
                temp_ent.append(ent)
        entities = temp_ent
    #Now we have a set of entites and we need to find all relations going from this relationship and also the final relationship
            #should be a a pert of the returned relationship
    #Find all the outgoing and incoming relationships
    outgoing_relationships = []
    incoming_relationships = []
    for ent in entities:
        rel = dbp.get_properties(ent,label=False)
        outgoing_relationships =  outgoing_relationships + list(set(rel[0]))
        incoming_relationships = incoming_relationships + list(set(rel[1]))
    outgoing_relationships = list(set(outgoing_relationships))
    incoming_relationships = list(set(incoming_relationships))
    return [outgoing_relationships,incoming_relationships]

def second_top_k_relation(entity,question,relations,k=5,scoring_method = 1):
    entity_label = dbp.get_label(entity)
    temp_relations = []
    for rel_tup in relations:
        if rel_tup[0] not in black_list and rel_tup[3] not in black_list:
            if scoring_method == 1:
                # Find the similarity between the ent+rel and the question
                phrase_1 = entity_label + " " + nlutils.get_label_via_parsing(rel_tup[0]) + " " + nlutils.get_label_via_parsing(rel_tup[3])
                similarity_score = sim.phrase_similarity(phrase_1, question)
                temp_relations.append((rel_tup[0], rel_tup[1],rel_tup[2],rel_tup[3], similarity_score*rel_tup[2]))
            if scoring_method == 2:
                # Find the similarity between the ent+rel and the question
                phrase_1 = nlutils.get_label_via_parsing(rel_tup[3])
                similarity_score = sim.phrase_similarity(phrase_1, question)
                temp_relations.append((rel_tup[0], rel_tup[1],rel_tup[2],rel_tup[3], similarity_score*rel_tup[2]))
            if scoring_method == 3:
                phrase_1 = nlutils.get_label_via_parsing(rel_tup[3])
                similarity_score_1 = sim.phrase_similarity(phrase_1, question)
                phrase_1 = entity_label + " " + nlutils.get_label_via_parsing(rel_tup[0]) + " " + nlutils.get_label_via_parsing(rel_tup[3])
                similarity_score_2 = sim.phrase_similarity(phrase_1, question)
                temp_relations.append((rel_tup[0], rel_tup[1], rel_tup[2], rel_tup[3], similarity_score_1*rel_tup[2]*similarity_score_2))
        else:
            continue
    temp_relations = sorted(temp_relations, key=lambda tup: tup[4], reverse=True)
    if len(temp_relations) > k:
        return temp_relations[:k]
    else:
        return temp_relations

def run(_readfiledir='../data/preprocesseddatasample/', _writefilename='resources/parsed_data.json',debug=False):
    """
    Get the show on the road.

    :param _readfiledir:   the filename (directory info included) to read the JSONs that need parsing
    :param _writefilename:  the file to which the parsed (embedded+padded) data is to be written to
    :param _debug:          the boolean param can be overwritten if wanted.
    :return: statuscode(?)
    """

    # Create vars to keep ze data @TODO: think of datatype here
    data_embedded = []

    # Load the vectorizing matrices in memory. TAKES TIME. Prepare your coffee now.
    # prepare("GLOVE")

    # Read JSON files.
    correct_counter = 0
    processed_counter = 0
    for filename in os.listdir(_readfiledir):
        data = json.load(open(os.path.join(_readfiledir, filename)))

        # Each file has multiple datapoints (questions).
        for question in data:


            # Collect the repsonse
            # v_q = question_vector(question['corrected_question'])
            entity_list = question['entity']
            # print question['path']
            path_length = len(question['path'])
            # print path_length
            temp_correct_path = [str(a[1:]) for a in question['path']] #removes the sign in front of each relations.
            if len(entity_list) > 1:
                if DEBUG:
                    print "cannot handel it right now"
                continue
            #Find a list of all outgoing and incoming relationships.
            right_properties, left_properties = dbp.get_properties(_uri=entity_list[0],label=False)
            relations = [(rel,'outgoing') for rel in right_properties]
            relations.extend([(rel,'incoming') for rel in left_properties])
            relations = top_k_relation(entity_list[0],question['corrected_question'],relations)
            print question['sparql_query']
            if path_length == 1:
                # print "the selected relations for the first hop are"
                # pprint(relations)
                temp_path = [relations[0][0]]
                print temp_path,temp_correct_path
                if temp_path == temp_correct_path:
                    correct_counter = correct_counter + 1
            chain_relations = []
            if path_length == 2:
                for rel in relations:
                    next_relations = updated_get_relationship_hop(entity_list[0],[rel])
                    for next_relation in next_relations:
                        chain_relations = chain_relations + [(rel[0],rel[1],rel[2],r) for r in next_relation]
                chain_relations = second_top_k_relation(entity_list[0],question['corrected_question'],chain_relations)
                # print("the second hop relations are")
                # pprint(chain_relations)
                temp_path = [chain_relations[0][0],chain_relations[0][3]]
                print temp_path,temp_correct_path
                if temp_path == temp_correct_path:
                    correct_counter = correct_counter + 1
            processed_counter = processed_counter + 1
            print correct_counter,processed_counter
            # raw_input()

if __name__ == "__main__":
    run()
