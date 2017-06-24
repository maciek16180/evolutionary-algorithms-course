import numpy as np
from cost import draw_individual_array as draw_individual
from cost import cost
from matplotlib.pyplot import imshow
import pickle


class PBILc():
    
    def __init__(self, pop_size, learning_rate, std_init, std_decay, ref_img,
                 F='default', n_best=2, n_worst=1, num_edges=3, num_figs=50, mode='polygon_simple'):
        
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
        
        self.learning_rate = learning_rate
        self.pop_size = pop_size
        
        self.perms = np.array([np.random.permutation(self.num_figs) for i in xrange(self.pop_size)])
        
        self.n_best = n_best
        self.n_worst = n_worst
        self.std_init = std_init
        self.std_decay = std_decay
        self.ref_img = ref_img
        self.ref_matrix = np.array(self.ref_img, dtype=np.int32)
        self.F = lambda i: -F(self.population[i].reshape(self.num_figs, self.fig_len)[self.perms[i]], self.ref_img, 
                              mode=self.mode, ref_matrix=self.ref_matrix)
        self.log = []
        self.iterations_done = 0
        self.best_imgs = []
        
        self.sigmas = np.zeros(self.chrom_len) + self.std_init
        self.mus = np.random.rand(self.chrom_len)
        
        self.population = self._generate_population()
        self.scores = self._score_population()
        
        self.best_ind = (self.population[self.scores.argmax()], self.perms[self.scores.argmax()])
        self.best_ind_score = self.scores.max()
        
    def _generate_population(self):
        pop = np.random.normal(self.mus, self.sigmas, (self.pop_size, self.chrom_len))
        pop[pop < 0] = 0
        pop[pop > 1] = 1
        self.perms = random_swap(self.perms, .8)
        return pop
    
    def _score_population(self):
        return np.array(map(self.F, xrange(self.pop_size)))
        
    def _one_iteration(self):
        best_indices = self.scores.argpartition(-self.n_best)[-self.n_best:]
        
        # update the best individual ever
        best_scores = self.scores[best_indices]
        best_index = best_scores.argmax()        
        if best_scores[best_index] > self.best_ind_score:
            self.best_ind_score = best_scores[best_index]
            self.best_ind = (self.population[best_indices][best_index], self.perms[best_indices][best_index])
        
        worst_indices = self.scores.argpartition(self.n_worst)[:self.n_worst]
        
        positive = self.population[best_indices].sum(axis=0)
        negative = self.population[worst_indices].sum(axis=0)
        
        self.mus *= 1 - self.learning_rate
        self.mus += self.learning_rate * (positive - negative)
        
        self.mus[self.mus > 1] = 1
        self.mus[self.mus < 0] = 0
        
        topK = self.scores.argpartition(-self.pop_size / 5)[-self.pop_size / 5:]
        
        if self.std_decay is not None:
            self.sigmas *= self.std_decay
        else:
            self.sigmas = self.population[topK].std(axis=0)
        
        self.population = self._generate_population()
        self.scores = self._score_population()
    
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
        name = name + '_pbilc_%i_%.3f_%.3f_%.3f_%i_%i_%i' % (self.pop_size, self.learning_rate, self.std_init, self.std_decay,
                                                             self.num_figs, self.num_edges, self.iterations_done)
        with open(name + '.pkl', 'w') as f:
            pickle.dump(self.log, f)
            
        self.best_imgs[-1].save(name + '.png', 'png')
                                      
    def best_img(self):
        return draw_individual(self.best_ind.reshape(self.num_figs, self.fig_len), self.ref_img.size, mode=self.mode)
    