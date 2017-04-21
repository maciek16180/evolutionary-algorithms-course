import numpy as np
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

data_path = 'heldkarp-amp/data/TSP/'


def prepare_tsp(fname):
    dists = load_tsp(fname)
    opt = load_opt(fname)
    opt_sol = tsp_eval(opt[np.newaxis], dists)[0] if opt is not None else None

    print 'Optimal solution for {}: {}'.format(fname, opt_sol if opt_sol is not None else 'unknown')
    
    return dists, opt_sol

def plot_tsp(log, opt):
    plt.figure(figsize=(15,8))
    plt.plot(xrange(len(log)), log)
    if opt is not None:
        plt.plot(xrange(len(log)), [-opt]*len(log))
    plt.show()

def load_tsp(filename):
    with open(data_path + filename + '.tsp', 'r') as f:
        
        for line in f:
            if line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split()[-1]
                break
        
        if edge_weight_type == 'EXPLICIT':
            edge_weight_section = False
            pad = 1
            matrix = []
            for line in f:
                if edge_weight_section:
                    row = [0.] * pad + map(np.float32, line.split())
                    matrix.append(row)
                    pad += 1
                    if pad == len(row):
                        matrix.append(np.zeros(pad))
                        matrix = np.array(matrix)
                        return matrix + matrix.T
                if line.startswith('EDGE_WEIGHT_SECTION'):
                    edge_weight_section = True
                    
        elif edge_weight_type == 'EUC_2D':
            node_coord_section = False
            points = []
            for line in f:
                if node_coord_section:
                    if line.startswith('EOF'):
                        points = np.array(points)
                        return cdist(points, points)
                    points.append(line.split()[1:])
                if line.startswith('NODE_COORD_SECTION'):
                    node_coord_section = True
                
def load_opt(filename):
    filename = data_path + filename + '.opt.tour'
    if not os.path.isfile(filename):
        return None
    with open(filename, 'r') as f:
        tour_section = False
        opt = []
        for line in f:
            if tour_section:
                num = int(line.split()[0])
                if num == -1:
                    return np.array(opt) - 1
                opt.append(num)
            if line.startswith('TOUR_SECTION'):
                tour_section = True
                
def random_tsp_solution(dists, N):
    P = []
    for i in xrange(N):
        P.append(np.random.permutation(dists.shape[0]))
    P = np.array(P)
    score = tsp_eval(P, dists)
    best = score.argmin()
    return P[best], score[best]

def tsp_eval(P, dists):
    P_roll = np.roll(P, 1, axis=1)
    return dists[P, P_roll].sum(axis=1)

def dummy_eval(P):
    return np.ones(P.shape[0])