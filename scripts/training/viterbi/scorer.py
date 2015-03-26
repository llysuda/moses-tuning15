'''
Created on Mar 24, 2015

@author: lly
'''

class Scorer(object):
    '''
    classdocs
    '''


    def __init__(self, name):
        '''
        Constructor
        '''
        self._num_stat = -1
        pass
    
    def numOfStat(self):
        return self._num_stat
    
    def score(self, stat):
        pass
    
    def inner(self, list1, list2):
        assert(len(list1) == len(list2))
        
        total = [w1+w2 for w1, w2 in zip(list1,list2)]
        #total = float(0)
        #for w1, w2 in zip(list1, list2):
        #    total += w1 * w2
        return sum(total)
