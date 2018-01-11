''' Takes a dbpedia entity and finds all the outgoing and incomoing relationships. '''

# Importing internal classes/libraries
import utils.dbpedia_interface as db_interface
import utils.natural_language_utilities as nlutils
from pprint import pprint

# TODO: Add stopword relation list here.
def get_properties(_uri, _right=True, _left=True):
	dbp = db_interface.DBPedia(_verbose=True,caching=True)
	if _right:
		right_properties = list(set(dbp.get_properties_of_resource(_resource_uri = _uri)))
		right_properties = [nlutils.get_label_via_parsing(rel) for rel in right_properties]
	if _left:
		left_properties = list(set(dbp.get_properties_of_resource(_resource_uri = _uri, right= False)))
		left_properties = [nlutils.get_label_via_parsing(rel) for rel in left_properties]
	if _right and _left:
		return right_properties, left_properties
	elif _right:
		return right_properties
	else:
		return left_properties

if __name__ == "__main__":
	'''A small test to verify the system'''
	pprint(get_properties('http://dbpedia.org/resource/Doctor_Who'))


