#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO:
# µ,λ (trzeba z λ rodziców zrobić > µ,λ dzieci i wybrać z nich nowe µ)
# ograniczenia (wprowadzamy kary za nagięcie ograniczeń)
#

import numpy as np

'''
ES(µ+λ) and ES(µ,λ) algorithms

d  = chromosome length
F  = evaluation function
N  = population size
M  = number of parents in each iteration
k  = learning rate
num_iter = number of iterations
init_fn = initializer for the first population
init_std = initial sigmas

in comma:
    num_children = number of children in each iteration

returns: a tuple (s, log) where
    s   = best solution from the last population
    log = score statistics from all iterations
'''

def mutate(parents, tau, tau0):
    d = int(parents.shape[1] / 2)
    M = parents.shape[0]
    
    eps = np.random.normal(scale=tau, size=(M, d))
    eps0 = np.random.normal(scale=tau0, size=(M, 1))
    new_sigmas = parents[:, d:] * np.exp(eps + eps0)
    eps = np.random.normal(scale=new_sigmas)
    new_xs = parents[:, :d] + eps
    
    return np.hstack([new_xs, new_sigmas])

def es_plus(d, F, N, M, k, num_iter, init_fn, init_std=1, logging=False, log_interval=1, roulette=False):
    
    X = init_fn(d, N)
    S = np.ones((N,d), dtype=np.float32) * init_std
    P = np.hstack([X, S])
    
    tau = k / np.sqrt(2 * d)
    tau0 = k / np.sqrt(2 * np.sqrt(d))

    score = F(X)
    scores = []
    sigmas = []
    
    for i in xrange(num_iter):
        
        sigmas.append(P[:, d:].mean(axis=0))
        
        if logging and not i % log_interval:
            print 'Starting iteration {} out of {}...'.format(i + 1, num_iter)

        scores.append(np.array([score.min(), score.mean(), score.max()]))
        
        if logging and not i % log_interval:
            print 'Current population stats: \t{:.10f}, \t{:.10f}, \t{:.10f}'.format(*scores[-1])
            
        if not roulette:
            parents = P[score.argpartition(-M)[-M:]]
        else:
            #score_positive = -1 / score
            score_positive = score - score.min()
            parent_probs = score_positive / score_positive.sum()
            parents = P[np.random.choice(N, p=parent_probs, size=M)]
            
        children = mutate(parents, tau, tau0)
        
        P = np.vstack([P, children])
        score = F(P[:, :d])
        best = score.argpartition(-N)[-N:]
        P = P[best]
        score = score[best]
        
    return P[np.argmax(score)][:d], np.vstack(scores), np.vstack(sigmas)

def mutate_comma(parents, tau, tau0, num_children): # deterministic version
    d = int(parents.shape[1] / 2)
    num_repeats = int(num_children / parents.shape[0])
    parents = np.repeat(parents, num_repeats, axis=0)
    
    eps = np.random.normal(scale=tau, size=(num_children, d))
    eps0 = np.random.normal(scale=tau0, size=(num_children, 1))
    new_sigmas = parents[:, d:] * np.exp(eps + eps0)
    eps = np.random.normal(scale=new_sigmas)
    new_xs = parents[:, :d] + eps
    
    return np.hstack([new_xs, new_sigmas])

def es_comma(d, F, N, M, num_children, k, num_iter, init_fn, init_std=1, logging=False, log_interval=1, roulette=False):
    assert num_children >= N
    if not roulette:
        assert not num_children % M
    
    X = init_fn(d, N)
    S = np.ones((N,d), dtype=np.float32) * init_std
    P = np.hstack([X, S])
    
    tau = k / np.sqrt(2 * d)
    tau0 = k / np.sqrt(2 * np.sqrt(d))

    score = F(X)
    scores = []
    sigmas = []
    
    
    for i in xrange(num_iter):
        sigmas.append(P[:, d:].mean(axis=0))
        
        
        if logging and not i % log_interval:
            print 'Starting iteration {} out of {}...'.format(i + 1, num_iter)

        scores.append(np.array([score.min(), score.mean(), score.max()]))
        
        if logging and not i % log_interval:
            print 'Current population stats: \t{:.10f}, \t{:.10f}, \t{:.10f}'.format(*scores[-1])
            
        if not roulette:
            parents = P[score.argpartition(-M)[-M:]]        
            children = mutate_comma(parents, tau, tau0, num_children)
        else:
            # when we use roulette parent selection, M doesn't matter
            # score_positive = -1 / score
            score_positive = score - score.min()
            parent_probs = score_positive / score_positive.sum()
            parents = P[np.random.choice(N, p=parent_probs, size=num_children)]
            children = children = mutate(parents, tau, tau0)
        
        score = F(children[:, :d])
        best = score.argpartition(-N)[-N:]
        P = children[best]
        score = score[best]
        
    return P[np.argmax(score)][:d], np.vstack(scores), np.vstack(sigmas)

def plot_sol(log, opt, fname=None):
    plt.plot(xrange(len(log)), log)
    if opt is not None:
        plt.plot(xrange(len(log)), [-opt]*len(log))
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()