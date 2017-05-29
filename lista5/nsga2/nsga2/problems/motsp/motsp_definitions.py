import math
from nsga2 import seq
from nsga2.problems.problem_definitions import ProblemDefinitions
from scipy.spatial.distance import cdist
import numpy as np

class MOTSPDefinitions(ProblemDefinitions):

    def __init__(self, problem_name, path='data/'):
        
        self.n = 100
        
        coords0 = []
        coords1 = []
        
        with open(path + 'euclid' + problem_name[0] + '100.tsp') as f:
            for i in xrange(6):
                f.readline()
            for line in f:
                coords0.append(map(float, line.split()[1:]))
            
        with open(path + 'euclid' + problem_name[1] + '100.tsp') as f:
            for i in xrange(6):
                f.readline()
            for line in f:
                coords1.append(map(float, line.split()[1:]))
                
        assert len(coords0) == self.n and len(coords1) == self.n
        
        self.coords0 = np.array(coords0)
        self.coords1 = np.array(coords1)
        
        assert self.coords0.shape == (self.n, 2)
        
        self.dists0 = cdist(self.coords0, self.coords0)
        self.dists1 = cdist(self.coords1, self.coords1)            

    def f1(self, individual):
        inds = np.vstack([individual.features, individual.features[1:] + [individual.features[0]]])
        #inds = individual.inds
        return self.dists0[inds[0], inds[1]].sum()

    def f2(self, individual):
        inds = np.vstack([individual.features, individual.features[1:] + [individual.features[0]]])
        #inds = individual.inds
        return self.dists1[inds[0], inds[1]].sum()

    # not implemented
    def perfect_pareto_front(self):
        domain = seq(0, 1, 0.01)
        return domain, map(lambda x1: 1 - math.sqrt(x1), domain)
