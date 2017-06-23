import numpy as np
from cost import draw_individual_array as draw_individual
from cost import cost
from matplotlib.pyplot import imshow
from cx import cx_one_point, cx_random


class ES():
    
    def __init__(self, pop_size, num_children, learning_rate, std_init, ref_img, 
                 roulette=True, cx='default', F='default', num_edges=3, num_figs=50, mode='polygon_simple', version='plus'):
        
        assert version in ['plus', 'comma']
        
        if F == 'default':
            F = cost
            
        self.mode = mode
        self.version = version
        
        if self.mode == 'polygon_simple':
            self.num_edges = num_edges
            self.fig_len = 2 * num_edges + 4
        elif self.mode == 'circle':
            self.fig_len = 7 # x, y, r, RGBA
        self.num_figs = num_figs
        self.chrom_len = self.fig_len * self.num_figs
        
        self.roulette = roulette
            
        if cx in ['default', 'random']:
            cx = cx_random
        elif cx == 'one_point':
            cx = cx_one_point
        
        self.cx = None if cx is None else lambda P: cx(P, self.num_figs, self.fig_len)
        
        self.pop_size = pop_size
        self.num_children = num_children
        self.learning_rate = learning_rate
        self.std_init = std_init
        self.ref_img = ref_img
        self.ref_matrix = np.array(self.ref_img, dtype=np.int32)
        self.F = lambda x: -F(x[:self.chrom_len].reshape(self.num_figs, self.fig_len), self.ref_img, 
                              mode=self.mode, ref_matrix=self.ref_matrix)
        
        self.tau = self.learning_rate / np.sqrt(2 * self.chrom_len)
        self.tau0 = self.learning_rate / np.sqrt(2 * np.sqrt(self.chrom_len))
        
        self.log = []
        self.iterations_done = 0
        self.best_imgs = []
        self.log_sigmas = []
        
        self.population = self._initial_population()
        self.scores = self._score_population(self.population)
        
        self.best_ind = self.population[self.scores.argmax()]
        self.best_ind_score = self.scores.max()
        
    def _initial_population(self):
        X = np.random.rand(self.pop_size, self.chrom_len)
        S = np.zeros((self.pop_size, self.chrom_len), dtype=np.float32) + self.std_init
        return np.hstack([X, S])
    
    def _score_population(self, P):
        return np.array(map(self.F, P))
        
    def _one_iteration(self):
        
        if not self.roulette:
            parents = self.population[self.scores.argpartition(-self.num_children)[-self.num_children:]]
        else:
            score_positive = self.scores - self.scores.min()
            parent_probs = score_positive / float(score_positive.sum())
            M = self.num_children if self.cx is None else self.num_children / 2
            parents = self.population[np.random.choice(self.pop_size, p=parent_probs, size=M)]
            
            if self.cx is not None:
                parents2 = self.population[np.random.choice(self.pop_size, p=parent_probs, size=M)]
                parents = self.cx(np.concatenate([parents[:, np.newaxis], parents2[:, np.newaxis]], axis=1))                
            
        children = mutate(parents.copy(), self.tau, self.tau0, self.num_figs, self.fig_len)
        
        P = np.vstack([self.population, children])
        scores = np.hstack([self.scores, self._score_population(children)])
        best = scores.argpartition(-self.pop_size)[-self.pop_size:]
        self.population = P[best]
        self.scores = scores[best]
        
        best_ever_idx = self.scores.argmax()
        if self.scores[best_ever_idx] > self.best_ind_score:
            self.best_ind = self.population[best_ever_idx]
            self.best_ind_score = self.scores[best_ever_idx]
    
    def train(self, num_it, debug=None):
        for i in xrange(num_it):
            self._one_iteration()
            self.iterations_done += 1
            
            if debug is not None and not self.iterations_done % debug:                
                self.log_sigmas.append(self.population[:, self.chrom_len:].mean(axis=0))
                self.log.append((self.iterations_done, [self.scores.min(), int(self.scores.mean()),
                                 self.scores.max(), self.best_ind_score]))
                
                print 'Scores after %i iterations: %s' % (self.iterations_done, self.log[-1][1])
                self.best_imgs.append(self.best_img())
        imshow(self.best_imgs[-1])
                                      
    def best_img(self):
        return draw_individual(self.best_ind[:self.chrom_len].reshape(self.num_figs, self.fig_len), 
                               self.ref_img.size, mode=self.mode)

    
def mutate(population, tau, tau0, num_figs, fig_len):
    d = int(population.shape[1] / 2)
    M = population.shape[0]
    population = population.reshape(M, 2, num_figs, fig_len).transpose(0,2,1,3).reshape(M, num_figs, fig_len * 2)
    changed_polys = np.random.choice(num_figs, M)
    changed_params = np.random.choice(fig_len, M)
    
    eps = np.random.normal(scale=tau, size=(M))
    eps0 = np.random.normal(scale=tau0, size=(M))
    
    index_sig = np.arange(M), changed_polys, changed_params + fig_len
    index_x = np.arange(M), changed_polys, changed_params
    
    new_sigmas = population[index_sig] * np.exp(eps + eps0)
    eps = np.random.normal(scale=new_sigmas)
    new_xs = population[index_x] + eps
    
    new_xs[new_xs > 1] = 1
    new_xs[new_xs < 0] = 0
    
    population[index_sig] = new_sigmas
    population[index_x] = new_xs
    
    return population.reshape(M, 2, num_figs, fig_len).transpose(0,2,1,3).reshape(M, num_figs * fig_len * 2)

    
def mutate_orig(population, tau, tau0, *args):
    d = int(population.shape[1] / 2)
    M = population.shape[0]
    
    eps = np.random.normal(scale=tau, size=(M, d))
    eps0 = np.random.normal(scale=tau0, size=(M, 1))
    new_sigmas = population[:, d:] * np.exp(eps + eps0)
    eps = np.random.normal(scale=new_sigmas)
    new_xs = population[:, :d] + eps
    
    new_xs[new_xs > 1] = 1
    new_xs[new_xs < 0] = 0
    
    return np.hstack([new_xs, new_sigmas])


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
'''

in comma:
    num_children = number of children in each iteration

returns: a tuple (s, log) where
    s   = best solution from the last population
    log = score statistics from all iterations
'''

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
