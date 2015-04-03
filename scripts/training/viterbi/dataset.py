import random
from sympy.matrices.dense import randMatrix
from Crypto.Random.random import shuffle

class DataSet (object):

    def __init__(self):
        self.data = {}
        self._size = 0
        self._denseList = []
        pass
    
    def size(self):
        return len(self.data)
    
    def denseSize(self):
        return len(self._denseList)
    
    def score(self, weight, scorer):
        stats = [0] * scorer.numOfStat()
        for i in range(self.size()):
            best_j = 0
            model_score = float('-inf')
            for j in range(len(self.data[i])):
                cur_score = scorer.inner(weight, self.data[i][j][1])
                if cur_score > model_score:
                    best_j = j
                    model_score = cur_score
            
            stats = [x+y for x,y in zip(stats, self.data[i][best_j][0])]
        
        return scorer.score(stats)
    
    def samples(self):
        ret = set()
        #random.setstate(1234)
        keys = self.data.keys()
        #keys = random.shuffle(keys)
        for key in keys:
            size = len(self.data[key])
            ids = range(size)
            for iter in range(100):
                i = random.randint(0, size-1)
                j = random.randint(0, size-1)
                if i == j:
                    continue
                ret.add((key, i, j))
        ret = list(ret)
        random.shuffle(ret)
        
        return ret
            
    def Fvalues(self, samples, scorer):
        ret = []
        for sentId, i, j in samples:
            fvi = self.data[sentId][i][1]
            fvj = self.data[sentId][j][1]
            
            bleui = scorer.smooth_score(self.data[sentId][i][0])
            bleuj = scorer.smooth_score(self.data[sentId][j][0])
            
            if bleui <= bleuj:
                ret.append(fvi)
            ret.append(fvj)
            if bleui > bleuj:
                ret.append(fvi)
        return ret
        
    def load(self, scfiles, ffiles):
        
        fileCount = 0
        for scfile, ffile in zip(scfiles.split(','), ffiles.split(',')):
            scf = open(scfile)
            ffile = open(ffile)
            
            lineId = 0
            numOfLine = 0
            
            
            for scline, fline in zip(scf, ffile):
                scline = scline.strip()
                fline = fline.strip()
                # nbest for a new sentence
                if scline.startswith("SCORES_TXT_BEGIN_0"):
                    assert(fline.startswith("FEATURES_TXT_BEGIN_0"))
                   
                    sc_items = scline.split()
                    f_items = fline.split()
                    
                    lineId = int(sc_items[1])
                    numOfLine = int(sc_items[2])
                    
                    if lineId == 0 and fileCount == 0:
                        # init denseList
                        for f in f_items[4:4+int(f_items[3])]:
                            self._denseList.append(f)
                    
                elif scline.startswith("SCORES_TXT_END_0"):
                    continue
                else:
                    scores = [float(x) for x in scline.split()]
                    features = [float(x) for x in fline.split()]
                    
                    if lineId not in self.data:
                        self.data[lineId] = []
                    self.data[lineId].append((scores, features))
            fileCount += 1
