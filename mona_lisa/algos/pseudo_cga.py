import numpy as np
from cost import draw_individual_array as draw_individual
from cost import cost
from matplotlib.pyplot import imshow
import random


class PseudoCGA():
    
    def __init__(self, pop_size, num_children, ref_img,
                 F='default', num_edges=3, num_figs=50, mode='polygon_simple'):
        
        if F == 'default':
            F = cost
            
        self.mode = mode
        
        if self.mode == 'polygon_simple':
            self.num_edges = num_edges
            self.fig_len = 2 * num_edges + 4
        elif self.mode == 'circle':
            self.fig_len = 7 # x, y, r, RGBA
        self.num_figs = num_figs
        self.chrom_len = self.fig_len * self.num_figs
        
        self.pop_size = pop_size
        self.num_children = num_children
        self.ref_img = ref_img
        self.ref_matrix = np.array(self.ref_img, dtype=np.int32)
        self.F = lambda x: -F(x.reshape(self.num_figs, self.fig_len), self.ref_img, 
                              mode=self.mode, ref_matrix=self.ref_matrix)
        self.log = []
        self.iterations_done = 0
        self.best_imgs = []
        
        self.population = self._generate_population()
        self.scores = self._score_population(self.population)
        
        self.best_ind = self.population[self.scores.argmax()]
        self.best_ind_score = self.scores.max()
        
    def _generate_population(self):
        pop = np.random.rand(self.pop_size, self.chrom_len)
        return pop
    
    def _score_population(self, P):
        return np.array(map(self.F, P))
    
    def _make_children(self, P):
        M = P.shape[0]
        P = P.reshape(M, self.num_figs, self.fig_len)
        changed_polys = np.random.choice(self.num_figs, M)
        changed_params = np.random.choice(self.fig_len, M)
        
        index = np.arange(M), changed_polys, changed_params
        change = np.random.triangular(0, P[index], 1)        
        P[index] = change
        P = P.reshape(M, self.chrom_len)
        
        return P
        
    def _one_iteration(self):
        
        # update the best individual ever
        best_index = self.scores.argmax()        
        if self.scores[best_index] > self.best_ind_score:
            self.best_ind_score = self.scores[best_index]
            self.best_ind = self.population[best_index]
        
        children = self._make_children(self.population.repeat(self.num_children, axis=0))
        
        #print (children == self.population).all()
        
        P = np.vstack([self.population, children])
        scores = np.hstack([self.scores, self._score_population(children)])
        best = scores.argpartition(-self.pop_size)[-self.pop_size:]
        self.population = P[best]
        self.scores = scores[best]
    
    def train(self, num_it, debug=None):
        for i in xrange(num_it):
            self._one_iteration()
            self.iterations_done += 1
            
            if debug is not None and not self.iterations_done % debug:
                print 'Score after %i iterations: %i' % (self.iterations_done, self.best_ind_score)
                self.log.append((self.iterations_done, self.best_ind_score))
                self.best_imgs.append(self.best_img())
        imshow(self.best_imgs[-1])
        
    def save(self, name):
        name = name + '_pcga_%i_%i_%i_%i_%i' % (self.pop_size, self.num_children, self.num_figs, 
                                                self.num_edges, self.iterations_done)
        with open(name + '.pkl', 'w') as f:
            pickle.dump(self.log, f)
            
        self.best_imgs[-1].save(name + '.png', 'png')
                                          
    def best_img(self):
        return draw_individual(self.best_ind.reshape(self.num_figs, self.fig_len), self.ref_img.size, mode=self.mode)