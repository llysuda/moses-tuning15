import math
import sys

from scorer import Scorer

class BLEU (Scorer):
    
    def __init__(self, scconfig=""):
        self._num_stat = 9
        
    def score(self, stats):
        if len(filter(lambda x: x==0, stats)) > 0:
            return 0
        c, r = stats[-1], stats[1]
        bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[:-1:2],stats[1:-1:2])])
        return math.exp(min([0, 1-float(r)/c]) + 0.25 * bleu_prec)

