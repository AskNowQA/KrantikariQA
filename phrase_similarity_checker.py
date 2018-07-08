'''
    The file takes a relation and a entity as input and generates the ranking,
'''

import preProcessing
import utils.dbpedia_interface as db_interface
from pprint import pprint

dbp = db_interface.DBPedia(_verbose=True, caching=True)


def test(_entity, _relation):
    out, incoming = dbp.get_properties(_entity, _relation, label=False)
    rel = (_relation, True)
    rel_list = preProcessing.get_set_list(
        preProcessing.get_rank_rel([out, incoming], rel,score=True))
    pprint(rel_list)


test('http://dbpedia.org/property/assembly' ,'http://dbpedia.org/resource/Broadmeadows,_Victoria')