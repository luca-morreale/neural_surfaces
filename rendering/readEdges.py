
import numpy as np


def readEdges(filename):

    edges = []
    with open(filename, 'r') as stream:
        edge_group = []
        for line in stream.readlines():
           if line[0] == '-':
               edges.append(np.array(edge_group).astype(np.int32))
               edge_group = []
           else:
               edge_group.append([int(el) for el in line.split(' ')])

    return edges
