import numpy as np
from time import time
import pickle

problem = 'AB'

front = []

with open('data/best.euclid' + problem + '100.tsp') as f:
    for line in f:
        front.append(map(int, line.split()))
    
front = zip(*front)
        
    
def HV(reference_point):
    def calculate_hyper_volume(front):
        def volume(individual):
            hyper_cuboid_sides = []
            for i in range(len(reference_point)):
                side_length = abs(individual[i] - reference_point[i])
                hyper_cuboid_sides.append(side_length)
            return reduce(lambda x, y: x*y, hyper_cuboid_sides, 1)

        return reduce(lambda sum, individual: sum + volume(individual), front, 0) / len(front)
    return calculate_hyper_volume

print 'Optimal HV:', HV([200000, 200000])(zip(*front))


import sys

sys.path.append('../nsga2/')

from metrics.problems.motsp import MOTSPMetricsAB
from nsga2.evolution import Evolution
from nsga2.problems.motsp import MOTSP
from nsga2.problems.motsp.motsp_definitions import MOTSPDefinitions

def print_statistics(population, iteration):
    PF = population.fronts[0]
    metrics = MOTSPMetricsAB()
    print("Iteration %03d: HV = %0.2f, HVR = %0.2f" % (iteration, metrics.HV(PF), metrics.HVR(PF)))

statistics = []
def log_statistics(population, iteration):
    ranks = [individual.rank for individual in population]
    crowding_distances = [individual.crowding_distance for individual in population]
    objectives = [individual.objectives for individual in population]

    pareto_front = population.fronts[0]
    metrics = MOTSPMetricsAB()
    hv = metrics.HV(pareto_front)
    hvr = metrics.HVR(pareto_front)

    statistics.append((hv, hvr, ranks, crowding_distances, objectives))
    
    
#####################    
    
t0 = time()

pop_size = 200
num_it = 500

solver = Evolution(MOTSP(MOTSPDefinitions(problem)), num_it, pop_size)
solver.register_on_new_generation(print_statistics)
solver.register_on_new_generation(log_statistics)

results = solver.evolve()

print 'Time:', time() - t0

import pickle

with open('resultsAB200_500.pkl', 'w') as f:
    pickle.dump(results, f)
    
with open('statsAB200_500.pkl', 'w') as f:
    pickle.dump(statistics, f)