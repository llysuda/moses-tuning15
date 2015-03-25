'''
Created on Mar 24, 2015

@author: lly
'''

import logging

from bleu import BLEU

def GetScorer(sctype, scconfig):
    if sctype == "BLEU":
        return BLEU(scconfig)
    else:
        logging.info("unknown score type: " + sctype)
        raise Exception("unknown")
    return None

if __name__ == '__main__':
    pass