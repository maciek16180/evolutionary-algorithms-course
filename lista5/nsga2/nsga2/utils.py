"""NSGA-II related functions"""

import functools, random
from nsga2.population import Population
import numpy as np
from copy import deepcopy

class NSGA2Utils(object):
    
    def __init__(self, problem, num_of_individuals, mutation_strength=0.2, num_of_genes_to_mutate=5, num_of_tour_particips=2):
        
        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.mutation_strength = mutation_strength
        self.number_of_genes_to_mutate = num_of_genes_to_mutate
        self.num_of_tour_particips = num_of_tour_particips
        
    def fast_nondominated_sort(self, population):
        population.fronts = []
        population.fronts.append([]) 
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = set()
            
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.add(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                population.fronts[0].append(individual)
                individual.rank = 0
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i+1
                        temp.append(other_individual)
            i = i+1
            population.fronts.append(temp)
                    
    def __sort_objective(self, val1, val2, m):
        return cmp(val1.objectives[m], val2.objectives[m])
    
    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0
            
            for m in range(len(front[0].objectives)):
                front = sorted(front, cmp=functools.partial(self.__sort_objective, m=m))
                front[0].crowding_distance = self.problem.max_objectives[m]
                front[solutions_num-1].crowding_distance = self.problem.max_objectives[m]
                for index, value in enumerate(front[1:solutions_num-1]):
                    front[index].crowding_distance = (front[index+1].crowding_distance - front[index-1].crowding_distance) / \
                                                     (self.problem.max_objectives[m] - self.problem.min_objectives[m])
                
    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
           ((individual.rank == other_individual.rank) and \
            (individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1
    
    def create_initial_population(self):
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.problem.generateIndividual()
            self.problem.calculate_objectives(individual)
            population.population.append(individual)
            
        return population
    
    def create_children_old(self, population):
        children = []
        parents = []
        count = 0
        while count < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1.features == parent2.features:
                parent2 = self.__tournament(population)
                
            if self.problem.problem_type != 'MOTSP':
                child1, child2 = self.__crossover(parent1, parent2)
                self.__mutate(child1)
                self.__mutate(child2)
                children.append(child1)
                children.append(child2)
            else:
                parents.append(parent1)
                parents.append(parent2)
            
            count += 2
            
        if self.problem.problem_type == 'MOTSP':
            children = self.__crossover_tsp_pop(parents)
            for c in children:
                self.__mutate_tsp(c, 3)
        
        for c in children:
            self.problem.calculate_objectives(c)

        return children
    
    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1.features == parent2.features:
                parent2 = self.__tournament(population)
                
            if self.problem.problem_type == 'MOTSP':
                child1, child2 = self.__crossover_tsp(parent1, parent2)
            else:
                child1, child2 = self.__crossover(parent1, parent2)
                
            if self.problem.problem_type == 'MOTSP':
                self.__mutate_tsp_ls(child1)
                self.__mutate_tsp_ls(child2)
            else:
                self.__mutate(child1)
                self.__mutate(child2)
                
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)

        return children
    
    def __crossover_tsp_pop(self, individuals):
        pairs = np.array([x.features for x in individuals]).reshape(len(individuals) / 2, 2, -1)
        children = [self.problem.generateIndividual(False) for i in xrange(len(individuals))]
        
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
        
        children_features = pmx(pairs, 1.)
        
        for i in xrange(children_features.shape[0]):
            children[i].features = list(children_features[i])
            
        return children
    
    def __crossover_tsp(self, individual1, individual2):
        child1 = self.problem.generateIndividual(False)
        child2 = self.problem.generateIndividual(False)
        
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
        
        child1.features, child2.features = map(list, pmx(np.array([[individual1.features, individual2.features]]), 1.))
            
        return child1, child2
    
    def __crossover(self, individual1, individual2):
        child1 = self.problem.generateIndividual()
        child2 = self.problem.generateIndividual()
        genes_indexes = range(len(child1.features))
        half_genes_indexes = random.sample(genes_indexes, 1)
        for i in genes_indexes:
            if i in half_genes_indexes:
                child1.features[i] = individual2.features[i]
                child2.features[i] = individual1.features[i]
            else:
                child1.features[i] = individual1.features[i]
                child2.features[i] = individual2.features[i]
        return child1, child2
    
    # num_swaps random transpositions
    def __mutate_tsp(self, child, num_swaps=1):
        for i in xrange(num_swaps):
            idx0 = random.randint(0, len(child.features) - 1)
            idx1 = random.randint(0, len(child.features) - 1)
            child.features[idx0], child.features[idx1] = child.features[idx1], child.features[idx0]
            
    def __mutate_tsp_ls(self, child, k=500):
        P = np.array([child.features])
        
        def random_swap(P, tm, **kwargs):
            mutated_idx = np.random.binomial(1, tm, size=P.shape[0]).astype(np.bool)
            mut_size = mutated_idx.nonzero()[0].size
            if not mut_size:
                return P
            mut = np.random.choice(P.shape[1], size=(mut_size, 2))
            idx = np.indices((P.shape[0], 2))[0][mutated_idx]
            P[idx, mut] = P[idx, mut[:, [1,0]]]
            return P
        
        def local_search(P, tm, k=None):
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
            neigh = map(list, neigh.reshape(-1, perm_len))
            neigh.append(list(P[0]))
            
            cands = []
            for i in xrange(k + 1):
                c = self.problem.generateIndividual(False)
                c.features = neigh[i]
                self.problem.calculate_objectives(c)
                cands.append(c)
                
            pop = Population()
            pop.extend(cands)
            self.fast_nondominated_sort(pop)
            
            return np.array([self.__tournament_tsp_all(pop).features])

        def local_search_plus(P, tm, k=None, it=5):
            for i in xrange(it):
                P = local_search(P, tm, k)
            return P
        
        def local_search_plus_swap(P, tm, k=None, it=5, tm_swap=.8, num_swaps=1):
            for i in xrange(it):
                P = local_search(P, tm, k)
                for j in xrange(num_swaps):
                    P = random_swap(P, tm_swap)
            return local_search(P, tm, k)   
        
        P = local_search_plus(P, 1., k=k)
        child.features = list(P[0])

    def __mutate(self, child):
        genes_to_mutate = random.sample(range(0, len(child.features)), self.number_of_genes_to_mutate)
        for gene in genes_to_mutate:
            child.features[gene] = child.features[gene] - self.mutation_strength/2 + random.random() * self.mutation_strength
            if child.features[gene] < 0:
                child.features[gene] = 0
            elif child.features[gene] > 1:
                child.features[gene] = 1
        
    def __tournament(self, population):
        participants = random.sample(population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or self.crowding_operator(participant, best) == 1:
                best = participant

        return best
    
    def __tournament_tsp_all(self, population):
        best = None
        for participant in population:
            if best is None or self.crowding_operator(participant, best) == 1:
                best = participant

        return best
