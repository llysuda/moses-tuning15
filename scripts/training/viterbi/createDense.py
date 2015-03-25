#!/bin/env python
import sys


if __name__ == '__main__':
    
    files = sys.argv[1:]

    fnames = []
    values = []
    for i,fname in enumerate(files):
        f = open(fname)
        count = 0
        for line in f:
            line = line.strip()
            name, val = line.split()

            if i == 0:
                fnames.append(name)
            else:
                assert(fnames[count] == name)
            
            if i == 0:
                values.append(val)
            else:
                values[count] += " " + val

            count += 1

        f.close()
    
    # output

    for name, val in zip(fnames, values):
        print name + " " + val
        
