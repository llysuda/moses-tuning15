#!/usr/bin/env python

'''
Created on Mar 24, 2015

@author: lly
'''

import argparse
import logging
import random
import os.path

import numpy
import theano
#from future.backports.email.policy import default

from scorer import Scorer
import sfactory as ScorerFactory
from dataset import DataSet
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from blender import BlenderModel
#from future.backports.email.policy import default


def backtrace(i,j):
    trace = []
    while i >= 0 and j >= 0:
        trace.append(weights[i][j])
        j = matrix[i][j][0]
        i -= 1
    trace.reverse()
    return trace

def step((i,j)):
    #logging.info("processing %d, %d", i, j)
    # global weights, w, data, scorerInst, matrix
    best_father = -1
    best_bleu = 0
    
    im1 = i - 1
    for k in range(len(weights[i])):
        new_w = []
        if i > 0:
            # backtrace
            new_w = backtrace(im1,k)
        
        new_w.append(weights[i][j])
        for iw in w[i+1:]:
            new_w.append(iw)
        bleu = data.score(new_w, scorerInst)
        
        if bleu > best_bleu:
            best_father = k
            best_bleu = bleu
    return j, best_father, best_bleu
        

def ViterbiSearch():
    numpool = mp.cpu_count()
    
    for i in range(len(weights)):
        pool = ProcessPool(numpool)
        #results = []
        #for j in range(len(weights[i])):
        #    results.append(step((i, j)))
        results = pool.map(step, [(i,j) for j in range(len(weights[i]))])
        
        pool.close()
        pool.join()
        
        for j, father, bleu in results:
            matrix[i][j] = (father, bleu)
          
    
    
    # backtrace for best path
    #bleu = max(matrix[-1][:][1])
    #i,j = len(matrix)-1, matrix[-1][:][1].index(bleu)
    bleu = 0
    j = -1
    for idx in range(len(matrix[-1])):
        if bleu < matrix[-1][idx][1]:
            bleu = matrix[-1][idx][1]
            j = idx
    
    best_w = backtrace(len(weights)-1, j)
    #TODO
    
    return best_w, bleu
        

def loadWeights(file):
    names = []
    weights = []
    prev_name = ""
    count = 0
    with open(file) as wfile:
        for line in wfile:
            items = line.strip().split()
            name = items[0][:-1]
            values = [float(x) for x in items[1:]]
            
            if prev_name == name:
                if count == 0:
                    names[-1] += "_"+str(count)
                count += 1
                name += "_" + str(count)
            else:
                prev_name = name
                count += 0
            
            names.append(name)
            weights.append(values)
    return names, weights
            

def InitWeight(ifile):
    f = open(ifile,'r');
    #w = []
    line = f.readline().strip()
    w = [float(x) for x in line.split()]
    #for i, points in enumerate(weights):
    #    j = random.randint(0, len(points)-1)
    #    w.append(points[j])
    f.close()
    return w

def Output(file, w, names):
    if file != "":
        with open(file,'w') as f:
            count = 0
            #for aw, name in zip(w, names):
            #    f.write("F"+str(count)+" "+str(aw)+"\n")
            #    count += 1
            f.write(" ".join([str(x) for x in w]))
    else:
        print " ".join([str(x) for x in w])
        #count = 0
        #for aw, name in zip(w,names):
        #    print "F"+str(count)+" "+str(aw)
        #    count += 1
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    parser = argparse.ArgumentParser(description='Viterbi search for combining sets of weights')
    parser.add_argument('--sctype', help='score type: BLEU (default), METEOR, RED......', default="BLEU")
    parser.add_argument('--dense', help='dense file, one feature per line in format: name val1 val2 val3', required=True)
    parser.add_argument('--scconfig', help='config for scorer', default="")
    parser.add_argument('--scfile', help='score data file', required=True)
    parser.add_argument('--ffile', help='feature file', required=True)
    parser.add_argument('--iter', help='iteration', type=int, default=100)
    parser.add_argument('-o', '--out', help='output file',default="")
    parser.add_argument('-d', help='number of dense features', type=int)
    parser.add_argument('-n', help='the number of random points', type=int, default=20)
    parser.add_argument('--threads', help='num of threads, no used', type=int)
    parser.add_argument('--ifile', help='weight file')
    parser.add_argument('--batch', help='number of dense features', type=int, default=10)
    #parser.add_argument('--model', help='weight file')

    args = parser.parse_args()
    
    sctype = args.sctype
    scconfig = args.scconfig
    denseFile = args.dense
    scfile = args.scfile
    ffile = args.ffile
    iter = args.iter
    ofile = args.out
    N = args.n
    batch = args.batch
    
    # scorer
    scorerInst = ScorerFactory.GetScorer(sctype, scconfig)
    
    # load  data: scfile and ffile
    data = DataSet()
    data.load(scfile, ffile)
    
    # load feature
    wnames, weights = loadWeights(denseFile)
    
    shape = (len(weights), len(weights[0]))
    bm = BlenderModel(input_shape=shape, batch_size=batch)
    
    model = "model"
    if os.path.isfile(model):
        bm.Load(model)
    #weights = [list(x) for x in zip(*weights)]
    
    best_score = 0
    best_w = None
    for loop in range(1):
        #logging.info('random init ' + str(loop))
        
        init_w = bm.Weight(weights)#InitWeight(args.ifile)
        
        Init_score = data.score(init_w, scorerInst)
        
        logging.info("init BLEU = " + str(Init_score))
        
        #best_score = Init_score
        w = list(init_w)

        if Init_score > best_score:
            best_score = Init_score
            best_w = list(init_w)
            #Output(ofile, w, wnames)
        
        #matrix = [[(-1,0) for j in weights[i]]for i in range(len(weights))]
        
        prev_score = Init_score
        for it in range(1, iter+1):
            logging.info("\t iter " + str(it) + "...")
            
            samples = data.samples()
            total_cost = 0.0
            
            for i in range(0, len(samples), batch):
                if i+batch > len(samples):
                    continue
                one_batch = samples[i:i+batch]
                
                fvalues = data.Fvalues(one_batch, scorerInst)
                
                cost = bm.Train(weights, fvalues)
                total_cost += cost
                
                #logging.info("batch from index "+ str(i)+"/"+ str(len(samples)) +",  cost= " + str(cost))
            logging.info("\t\t total cost= " + str(cost))   
            w = bm.Weight(weights)
            curr_score = data.score(w, scorerInst)
            logging.info("\t\t BLEU = " + str(curr_score))
            if curr_score > best_score:
                best_score = curr_score
                best_w = list(w)
                bm.Save(model)
                #Output(ofile, w, wnames)

            if abs(prev_score - curr_score) < 0.000001:
                break

            prev_score = curr_score
            
            # clear matrix
            #matrix = [[(-1,0) for j in weights[i]]for i in range(len(weights))]
    logging.info("Best point: "+" ".join([str(x) for x in best_w])+ " => "+ str(best_score))

    f = open('weights.txt','w')
    f.write(" ".join([str(x) for x in best_w])+"\n")
    f.close()
    
    

    Output(ofile,best_w,wnames)
