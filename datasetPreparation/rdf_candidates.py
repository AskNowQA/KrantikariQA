'''
    The objective of the class is to find the rdf type candidates for the given path.
    It assumes only specific kind of SPARQL's defined in the macros below.
'''


#SPARQL templates supported.

sparql_template_1 = {
	"-" : 'SELECT DISTINCT ?uri WHERE {?uri <%(r1)s> <%(te1)s> . }',
	"+" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri . }'
}

sparql_template_2 = {
	"++" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?x . ?x <%(r2)s> ?uri  . }',
	"-+" : 'SELECT DISTINCT ?uri WHERE { ?x <%(r1)s> <%(te1)s> . ?x <%(r2)s> ?uri  . }',
	"--" : 'SELECT DISTINCT ?uri WHERE { ?x <%(r1)s> <%(te1)s> . ?uri <%(r2)s> ?x  . }',
	"+-" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?x . ?uri <%(r2)s> ?x  . }'
}

sparql_template_3 = {
	"+-" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri . <%(te2)s> <%(r2)s> ?uri  . }',
	"--" : 'SELECT DISTINCT ?uri WHERE { ?uri <%(r1)s> <%(te1)s> . <%(te2)s> <%(r2)s> ?uri  . }',
	"-+" : 'SELECT DISTINCT ?uri WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <%(te2)s>  . }',
    "++" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri .  ?uri <%(r2)s> <%(te2)s>  . }'
}

sparql_template_ask = {
	"+" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> <%(te2)s> . }'
}

x_const = '.  ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_x . } '
uri_const = '. ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_uri . } '
uri_x_const = '. ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_x . ' \
			  '?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?cons_uri . } '



def construct_sparql(path,topic_entity):

    if len(path) == 2:
        sparql_template = sparql_template_1[path[0]]
        sparql = sparql_template % {'te1':topic_entity[0],'r1':path[1]}
    else:
        sparql_key = path[0]+path[2]

        if sparql_key in sparql_template_2 and len(topic_entity) == 1:
            sparql = sparql_template_2[sparql_key] % {'te1':topic_entity[0],'r1': path[1], 'r2': path[3]}
        else:
            sparql = sparql_template_3[sparql_key] % {'te1': topic_entity[0], 'r1': path[1], 'r2': path[3], 'te2':topic_entity[1]}

    return sparql


def construct_sparql_with_constraints(sparql):
    sparql_x_constraints = sparql.replace('. }', x_const)
    sparql_x_constraints = sparql_x_constraints.replace('SELECT DISTINCT ?uri', 'SELECT DISTINCT ?cons_x')
    sparql_uri_constraints = sparql.replace('. }', uri_const)
    sparql_uri_constraints = sparql_uri_constraints.replace('SELECT DISTINCT ?uri', 'SELECT DISTINCT ?cons_uri')
    return sparql_x_constraints, sparql_uri_constraints

def shoot_sparql(sparql,dbp,x):
    '''

        Clean the output.
    :param sparql: sparql query which has rdf type constrain
    :param dbp: dbpedia interface object
    :param x: type constraint on which variable
    :return:
    '''
    answer = dbp.get_answer(sparql)
    if x:
        return answer['cons_x']
    else:
        return answer['cons_uri']

def is_not_blacklisted(predicate):
    '''

    :param predicate: a str of relation - 'http://dbpedia.org/ontology/Person',
    :return: True if it is either ontology or property

    @TODO: Check if 'dbpedia.org/property' is a correct filter
    '''
    if 'dbpedia.org/ontology' in predicate or 'dbpedia.org/property' in predicate:
        return True
    return False


def generate_rdf_candidates(path,topic_entity,dbp):
    '''

    :param path: the path for which the candidates needs to be generated
    :param topic_entity: a list of topic entity
    :return: candidates

    '''
    sparql = construct_sparql(path,topic_entity)
    sparql_x_constraints, sparql_uri_constraints = construct_sparql_with_constraints(sparql)
    if len(path) > 2 and len(topic_entity) == 1:
        x_const = shoot_sparql(sparql = sparql_x_constraints, dbp=dbp, x=True)
        x_const = [rel.decode("utf-8") for rel in x_const if is_not_blacklisted(rel.decode('utf-8'))]
    else:
        x_const = []
    uri_const = shoot_sparql(sparql = sparql_uri_constraints, dbp=dbp, x=False)
    uri_const = [rel.decode("utf-8") for rel in uri_const if is_not_blacklisted(rel.decode("utf-8"))]
    return x_const,uri_const


