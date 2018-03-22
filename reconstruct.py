'''
	This will re-construct SPARQL given core-chain and topic entites.
'''

sparql_template_1 = {
	"-" : 'SELECT DISTINCT ?uri WHERE {?uri <%(r1)s> <%(te1)s> }',
	"+" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri }'
}

sparql_template_2 = {
	"++" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?x . ?x <%(r2)s> ?uri  . }',
	"-+" : 'SELECT DISTINCT ?uri WHERE { ?x <%(r1)s> <%(te1)s> . ?x <%(r2)s> ?uri  . }',
	"--" : 'SELECT DISTINCT ?uri WHERE { ?x <%(r1)s> <%(te1)s> . ?uri <%(r2)s> ?x  . }',
	"+-" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?x . ?uri <%(r2)s> ?x  . }'
}

sparql_template_3 = {
	"++" : 'SELECT DISTINCT ?uri WHERE { <%(te1)s> <%(r1)s> ?uri . <te2> <%(r2)s> ?uri  . }',
	"--" : 'SELECT DISTINCT ?uri WHERE { ?uri <%(r1)s> <%(te1)s> . ?uri <%(r2)s> <te2>  . }'
}

def reconstruct(ent,relations):
	if relations[2].count('+') + relations[2].count('-') == 1:
		rel_length = 1
	else:
		rel_length = 2
	if len(ent) == 1:
		if rel_length == 1:
			sparql = sparql_template_1[relations[0]]
			return sparql % {'r1':relations[1:],'te1':ent}