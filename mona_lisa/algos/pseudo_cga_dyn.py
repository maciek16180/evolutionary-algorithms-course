import numpy as np
from cost import draw_individual_list as draw_individual
from cost import cost
from matplotlib.pyplot import imshow
import random, pickle
from itertools import chain
from copy import deepcopy, copy


class PseudoCGA_dyn():
    
    def __init__(self, pop_size, num_children, ref_img,
                 F='default', min_num_edges=3, max_num_edges=3, max_num_figs=50, mode='polygon_simple',
                 mut_add_pol=0.01, mut_add_point=0.005):
        
        if F == 'default':
            F = cost
            
        self.mode = mode
        self.shape = ref_img.size
        
        self.add_pol_mut = mut_add_pol
        self.add_point_mut = mut_add_point
        
        if self.mode == 'polygon_simple':
            self.max_num_edges = max_num_edges
            self.min_num_edges = min_num_edges
        elif self.mode == 'circle':
            raise NotImplementedError
        self.max_num_figs = max_num_figs
        
        self.pop_size = pop_size
        self.num_children = num_children
        self.ref_img = ref_img
        self.ref_matrix = np.array(self.ref_img, dtype=np.int32)
        self.F = lambda x: -F(x, self.ref_img, mode=self.mode, ref_matrix=self.ref_matrix)
        
        self.log = []
        self.iterations_done = 0
        self.best_imgs = []
        
        self.population = [([], []) for i in xrange(self.pop_size)] # starting individuals have 0 polygons
        self.scores = self._score_population(self.population)
        
        for i in xrange(self.pop_size):
            for j in xrange(self.max_num_figs):
                p, c = self._generate_figure()
                self.population[i][0].append(p)
                self.population[i][1].append(c)
        
        self.best_ind = self.population[self.scores.argmax()]
        self.best_ind_score = self.scores.max()
        
    def _generate_figure(self):
        pol = [self._generate_point() for i in xrange(self.min_num_edges)]
        col = [random.randint(0, 255) for i in xrange(4)]
        return (pol, col)
    
    def _generate_point(self):
        return [random.randint(0, self.shape[0]), random.randint(0, self.shape[1])]
    
    def _score_population(self, P):
        return np.array(map(self.F, P))
    
    def _mutate_population(self, P):
        return [self._mutate_individual(ind) for ind in P]
    
    def _mutate_individual(self, ind):
        pols, cols = ind
        num_pols = len(cols)
        add_point = np.array([])
        
        if num_pols:
            mut_pol = random.randint(0, num_pols - 1)
            mut_axis = random.randint(0, 2 * len(pols[mut_pol]) + 3)
            
            if mut_axis < 4:
                col = cols[mut_pol][:]
                col[mut_axis] = int(random.triangular(0, col[mut_axis], 255))
                cols = cols[:mut_pol] + [col] + cols[mut_pol + 1:]
            else:
                xy = mut_axis % 2
                pol = map(copy, pols[mut_pol])
                pol[mut_axis / 2 - 2][xy] = int(random.triangular(0, pol[mut_axis / 2 - 2][xy], self.shape[xy]))
                pols = pols[:mut_pol] + [pol] + pols[mut_pol + 1:]
            
            add_point = np.where(np.random.rand(num_pols) < self.mut_add_point)[0]
            if add_point.size:
                pols = [map(copy, pol) for pol in pols]
                for i in add_point:
                    if len(pols[i]) < self.max_num_edges:
                        pols[i].append(self._generate_point())
                    
        if num_pols < self.max_num_figs and random.random() < self.mut_add_pol:
            cols = map(copy, cols)
            if not add_point.size:
                pols = [map(copy, pol) for pol in pols]
            pol, col = self._generate_figure()
            pols.append(pol)
            cols.append(col)        
            
        return pols, cols
        
    def _one_iteration(self):
        
        # update the best individual ever
        best_index = self.scores.argmax()        
        if self.scores[best_index] > self.best_ind_score:
            self.best_ind_score = self.scores[best_index]
            self.best_ind = self.population[best_index]
        
        children = chain(*[self.population for i in xrange(self.num_children)])
        children = self._mutate_population(children)
        
        P = self.population + children
        scores = np.hstack([self.scores, self._score_population(children)])
        best = scores.argpartition(-self.pop_size)[-self.pop_size:]
        self.population = [P[i] for i in best]
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
        name = name + '_pcgad_%i_%i_%i_%i_%i' % (self.pop_size, self.num_children, self.max_num_figs, 
                                                 self.max_num_edges, self.iterations_done)
        with open(name + '.pkl', 'w') as f:
            pickle.dump(self.log, f)
            
        self.best_imgs[-1].save(name + '.png', 'png')
                                          
    def best_img(self):
        return draw_individual(self.best_ind, self.ref_img.size, mode=self.mode)