class NoEntitiesFound(Exception): pass
class NoPathsFound(Exception): pass
class BadParameters(Exception):
    def __init___(self,dErrorArguments):
        Exception.__init__(self,"Unexpected value of parmeter {0}".format(dErrorArguments))
        self.dErrorArguments = dErrorArguments