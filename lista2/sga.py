#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time

'''
mutations for SGA
'''

def random_swap(P, tm, **kwargs):
    mutated_idx = np.random.binomial(1, tm, size=P.shape[0]).astype(np.bool)
    mut_size = mutated_idx.nonzero()[0].size
    if not mut_size:
        return P
    mut = np.random.choice(P.shape[1], size=(mut_size, 2))
    idx = np.indices((P.shape[0], 2))[0][mutated_idx]
    P[idx, mut] = P[idx, mut[:, [1,0]]]
    return P

def local_search(P, tm, F, k=None):
    mutated_idx = np.random.binomial(1, tm, size=P.shape[0]).astype(np.bool)
    mut_size = mutated_idx.nonzero()[0].size
    perm_len = P.shape[1]
    if not mut_size:
        return P
    
    if k is None:
        mut = np.vstack([np.vstack(np.triu_indices(perm_len, 1)).T, np.array([0,0])])
    else:
        mut = np.random.choice(perm_len, size=(k, 2))
    mut = np.repeat(mut[np.newaxis], mut_size, axis=0)
    
    idx = np.indices((mut_size, k or mut.shape[1], 2))
    
    neigh = np.swapaxes(np.repeat(P[mutated_idx, :, np.newaxis], k or mut.shape[1], axis=2), 1, 2)
    neigh[idx[0], idx[1], mut] = neigh[idx[0], idx[1], mut[:, :, [1,0]]]
    best = F(neigh.reshape(-1, perm_len)).reshape(mut_size, -1)
    
    mutated = neigh[np.arange(mut_size), best.argmax(axis=1)]
    
    return np.vstack([mutated, P[~mutated_idx]])

def local_search_plus(P, tm, F, k=None, it=5):
    for i in xrange(it):
        P = local_search(P, tm, F, k)
    return P
                         
def local_search_plus_swap(P, tm, F, k=None, it=5, tm_swap=.5):
    for i in xrange(it):
        P = local_search(P, tm, F, k)
        P = random_swap(P, tm_swap)
    return local_search(P, tm, F, k)   

def lsp(it, k=None):
    return lambda *args, **kwargs: local_search_plus(*args, it=it, k=k, **kwargs)

def lsps(it, tm_swap, k=None):
    return lambda *args, **kwargs: local_search_plus_swap(*args, it=it, tm_swap=tm_swap, k=k, **kwargs)

'''
crossover operators for SGA
'''

def pmx(pairs, tc):
    changed = np.random.rand(pairs.shape[0]) < tc
    num_changed = changed.nonzero()[0].size
    parents = pairs[changed]
    not_parents = pairs[~changed].reshape(-1, pairs.shape[2])
    if not num_changed:
        return not_parents
    
    inds = np.sort(np.random.choice(pairs.shape[2], size=(num_changed, 2)), axis=1)
    
    def get_val(x, dic):
        while x in dic:
            x = dic[x]
        return x
    
    C = []
    
    for i in xrange(num_changed):
        beg_idx, end_idx = inds[i]
        middle = slice(beg_idx, end_idx + 1)
        
        p1, p2 = parents[i]
        rev_parents = np.array([p2, p1])
        
        kids = np.zeros_like(parents[i])
        kids[:, middle] = parents[i][:, middle]
        kids[:, :beg_idx] = rev_parents[:, :beg_idx]
        kids[:, end_idx+1:] = rev_parents[:, end_idx+1:]
            
        def fix_child(c, p1, p2):
            temp = dict(zip(p1[middle], p2[middle]))
            middle_set = set(p1[middle]) - set(p2[middle])
            
            for i in middle_set:
                c[p2==i] = get_val(i, temp)
                
        fix_child(kids[0], p1, p2)
        fix_child(kids[1], p2, p1)
            
        C.append(kids)
        
    C.append(not_parents)
        
    return np.vstack(C)

'''
SGA algorithm for permutations

d  = chromosome length
F  = evaluation function
N  = population size
M  = number of parents in one iteration
tc = crossover probability
tm = mutation probability
num_iter = number of iterations

returns: a pair (s, log) where
    s   = best solution from the last population
    log = score statistics for all iterations
'''

def sga(d, F, N, M, tc, tm, num_iter, mutation=random_swap, crossover_fn=pmx, logging=False, init=None, whole_pop=False,
        log_interval=1):
    if init is not None:
        P = init
    else:
        P = []
        for i in xrange(N):
            P.append(np.random.permutation(d))
        P = np.array(P)

    score = F(P)
    scores = []

    for i in xrange(num_iter):

        if logging and not i % log_interval:
            print 'Starting iteration {} out of {}...'.format(i + 1, num_iter)

        scores.append(np.array([score.min(), score.mean(), score.max()]))
        parent_probs = score / score.sum()
        parents = P[np.random.choice(N, p=parent_probs, size=M)]
        pairs = parents[np.random.permutation(M)].reshape(-1, 2, d)
                 
        children = mutation(crossover_fn(pairs, tc), tm, F=F)
        
        # (µ + λ) replacement
        
        P = np.concatenate([P, children], axis=0)         
        score = F(P)
        best = score.argpartition(-N)[-N:]
        P = P[best]
        score = score[best]

        if logging and not i % log_interval:
            print 'Current population stats: \t{:.3f} \t{:.3f} \t{:.3f}'.format(*scores[-1])

    return P[np.argmax(score)] if not whole_pop else P, np.vstack(scores)