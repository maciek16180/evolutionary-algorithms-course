import numpy as np

'''
PBIL algorithm

d  = chromosome length
F  = evaluation function
N  = population size
t1 = learning rate
t2 = mutation probability
t3 = mutation intensity
num_iter = number of iterations
sigma = std dev

returns: a tuple (s, log, p) where
    s   = best solution from the last population
    log = score statistics from all iterations
    p   = probability model
'''

def pbil_weighted(d, F, N, t1, t2, t3, num_iter, sigma, logging=False):
    p = np.ones(d)
    P = np.random.normal(p, sigma, (N,d))
    score = F(P)
    scores = []
    
    for i in xrange(num_iter):
        
        if logging:
            print 'Starting iteration {} out of {}...'.format(i + 1, num_iter)
            
        best = P[np.argmax(score)]     
        scores.append(np.array([score.min(), score.mean(), score.max()]))
        
        if logging:
            print 'Current population stats: \t{:.3f}, \t{:.3f}, \t{:.3f}'.format(*scores[-1])
        
        p = p * (1 - t1) + best * t1
        mut = np.random.binomial(1, t2, d).astype(np.bool)
        p[mut] = p[mut] * (1 - t3) + np.random.binomial(1, .5) * t3
        P = np.random.normal(p, sigma, (N,d))
        score = F(P)
        
    return P[np.argmax(score)], np.vstack(scores), p
