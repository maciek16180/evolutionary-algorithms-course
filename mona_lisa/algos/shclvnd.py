import numpy as np
from cost import draw_individual_array as draw_individual
from cost import cost
from matplotlib.pyplot import imshow


class SHCLVND():
    
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
        self.n_best = n_best
        self.n_worst = n_worst
        self.std_init = std_init
        self.std_decay = std_decay
        self.ref_img = ref_img
        self.ref_matrix = np.array(self.ref_img, dtype=np.int32)
        self.F = lambda x: -F(x.reshape(self.num_figs, self.fig_len), self.ref_img, 
                              mode=self.mode, ref_matrix=self.ref_matrix)
        self.log = []
        self.iterations_done = 0
        self.best_imgs = []
        
        self.sigmas = np.zeros(self.chrom_len) + self.std_init
        self.mus = np.random.rand(self.chrom_len)
        
        self.population = self._generate_population()
        self.scores = self._score_population()
        
        self.best_ind = self.population[self.scores.argmax()]
        self.best_ind_score = self.scores.max()
        
    def _generate_population(self):
        pop = np.random.normal(self.mus, self.sigmas, (self.pop_size, self.chrom_len))
        pop[pop < 0] = 0
        pop[pop > 1] = 1
        return pop
    
    def _score_population(self):
        return np.array(map(self.F, self.population))
        
    def _one_iteration(self):
        best_indices = self.scores.argpartition(-self.n_best)[-self.n_best:]
        
        # update the best individual ever
        best_scores = self.scores[best_indices]
        best_index = best_scores.argmax()        
        if best_scores[best_index] > self.best_ind_score:
            self.best_ind_score = best_scores[best_index]
            self.best_ind = self.population[best_indices][best_index]
        
        worst_indices = self.scores.argpartition(self.n_worst)[:self.n_worst]
        
        positive = self.population[best_indices].mean(axis=0) - self.mus
        negative = self.population[worst_indices].mean(axis=0) - self.mus
        
        self.mus += self.learning_rate * positive
        self.mus -= self.learning_rate * negative
        
        self.mus[self.mus > 1] = 1
        self.mus[self.mus < 0] = 0
        
        self.sigmas *= self.std_decay
        
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
                                      
    def best_img(self):
        return draw_individual(self.best_ind.reshape(self.num_figs, self.fig_len), self.ref_img.size, mode=self.mode)
    

###########################################################

class SHCLVND_perm():
    
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
        
        positive = self.population[best_indices].mean(axis=0) - self.mus
        negative = self.population[worst_indices].mean(axis=0) - self.mus
        
        self.mus += self.learning_rate * positive
        self.mus -= self.learning_rate * negative
        
        self.mus[self.mus > 1] = 1
        self.mus[self.mus < 0] = 0
        
        self.sigmas *= self.std_decay
        
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
                                      
    def best_img(self):
        return draw_individual(self.best_ind[0].reshape(self.num_figs, self.fig_len)[self.best_ind[1]], 
                               self.ref_img.size, mode=self.mode)
    
    
def random_swap(P, tm):
    mutated_idx = np.random.binomial(1, tm, size=P.shape[0]).astype(np.bool)
    mut_size = mutated_idx.nonzero()[0].size
    if not mut_size:
        return P
    mut = np.random.choice(P.shape[1], size=(mut_size, 2))
    idx = np.indices((P.shape[0], 2))[0][mutated_idx]
    P[idx, mut] = P[idx, mut[:, [1,0]]]
    return P
    
###########################################################
    
    
class SHCLVND_dynamic():
    
    def __init__(self, pop_size, learning_rate, std_init, std_decay, ref_img, max_it_wo_improvement,
                 F='default', n_best=2, n_worst=1, num_edges=3, max_num_figs=50, mode='polygon_simple'):
        
        if F == 'default':
            F = cost
            
        self.mode = mode
        
        if self.mode == 'polygon_simple':
            self.num_edges = num_edges
            self.fig_len = 2 * num_edges + 4
        elif self.mode == 'circle':
            self.fig_len = 7 # x, y, r, RGBA
        self.num_figs = 1
        self.max_num_figs = max_num_figs
        self.chrom_len = self.fig_len * self.num_figs
        
        self.max_it_wo_improvement = max_it_wo_improvement
        self.timer = 0
        
        self.learning_rate = learning_rate
        self.pop_size = pop_size
        self.n_best = n_best
        self.n_worst = n_worst
        self.std_init = std_init
        self.std_decay = std_decay
        self.ref_img = ref_img
        self.ref_matrix = np.array(self.ref_img, dtype=np.int32)
        self.F = lambda x: -F(x.reshape(self.num_figs, self.fig_len), self.ref_img, 
                              mode=self.mode, ref_matrix=self.ref_matrix)
        self.log = []
        self.iterations_done = 0
        self.best_imgs = []
        
        self.sigmas = np.zeros(self.chrom_len) + self.std_init
        self.mus = np.random.rand(self.chrom_len)
        
        self.population = self._generate_population()
        self.scores = self._score_population()
        
        self.best_ind = self.population[self.scores.argmax()]
        self.best_ind_score = self.scores.max()
        
    def _generate_population(self):
        pop = np.random.normal(self.mus, self.sigmas, (self.pop_size, self.chrom_len))
        pop[pop < 0] = 0
        pop[pop > 1] = 1
        return pop
    
    def _score_population(self):
        return np.array(map(self.F, self.population))
        
    def _one_iteration(self, debug):
        best_indices = self.scores.argpartition(-self.n_best)[-self.n_best:]
        
        # update the best individual ever
        best_scores = self.scores[best_indices]
        best_index = best_scores.argmax()        
        if best_scores[best_index] > self.best_ind_score:
            self.best_ind_score = best_scores[best_index]
            self.best_ind = self.population[best_indices][best_index]
        
        worst_indices = self.scores.argpartition(self.n_worst)[:self.n_worst]
        
        positive = self.population[best_indices].mean(axis=0) - self.mus
        negative = self.population[worst_indices].mean(axis=0) - self.mus
        
        self.mus += self.learning_rate * positive
        self.mus -= self.learning_rate * negative
        
        self.mus[self.mus > 1] = 1
        self.mus[self.mus < 0] = 0
        
        self.sigmas *= self.std_decay
        
        if self.log and abs(self.log[0][0] - self.log[-1][0]) >= self.max_it_wo_improvement and \
                self.log[-self.max_it_wo_improvement / debug][1] == self.log[-1][1] and self.timer <= 0 and \
                self.num_figs < self.max_num_figs:
            print 'Adding figure. Figure count: %i' % (self.num_figs + 1)
            self.add_fig()
        
        self.population = self._generate_population()
        self.scores = self._score_population()
        
    def add_fig(self):
        self.sigmas /= self.std_decay**self.max_it_wo_improvement
        self.sigmas = np.hstack([self.sigmas, np.zeros(self.fig_len) + self.std_init])
        self.mus = np.hstack([self.mus, np.random.rand(self.fig_len)])
        self.num_figs += 1
        self.chrom_len += self.fig_len
        self.timer = self.max_it_wo_improvement
    
    def train(self, num_it, debug=10):
        for i in xrange(num_it):
            self._one_iteration(debug)
            self.iterations_done += 1
            self.timer -= 1
            
            if debug is not None and not self.iterations_done % debug:
                print 'Score after %i iterations: %i' % (self.iterations_done, self.best_ind_score)
                self.log.append((self.iterations_done, self.best_ind_score))
                self.best_imgs.append(self.best_img())
                imshow(self.best_imgs[-1])
                                      
    def best_img(self):
        return draw_individual(self.best_ind.reshape(-1, self.fig_len), self.ref_img.size, mode=self.mode)

    