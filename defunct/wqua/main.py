'''

This files attempts to create question answering just using word embedding.
'''

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
                if EMBEDDING == "GLOVE": token_embedding = embedding_glove[token]
                elif EMBEDDING == 'WORD2VEC': token_embedding = embedding_word2vec[token]

            except KeyError:
                if _report_unks: unks.append(token)
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



def run(_readfiledir='data/preprocesseddata/', _writefilename='resources/parsed_data.json',debug=False):
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
    prepare("GLOVE")

    # Read JSON files.
    for filename in os.listdir(_readfiledir):
        data = json.load(open(os.path.join(_readfiledir, filename)))

        # Each file has multiple datapoints (questions).
        for question in data:

            # Collect the repsonse
            v_q, v_tp, v_fps, v_y = parse(question)

            # Collect data for each question
            data_embedded.append([v_q, v_tp, v_fps, v_y])

    """
        ||TEMP||

        Pickle this data.
        Play around with it.
    """
    f = open('resources/tmp.pickle', 'w+')
    pickle.dump(data_embedded, f)


