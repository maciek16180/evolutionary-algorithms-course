import math
from nsga2 import seq
from nsga2.problems.problem_definitions import ProblemDefinitions

class ZDT4Definitions(ProblemDefinitions):

    def __init__(self):
        self.n = 30

    def f1(self, individual):
        #if any(abs(x) > 5 for x in individual.features):
        #    return 100000000000
        return individual.features[0]

    def f2(self, individual):
        #if any(abs(x) > 5 for x in individual.features):
        #    return 100000000000
        g = 1. + 10*(self.n - 1) + sum(x**2 - 10*math.cos(4*math.pi*x) for x in individual.features[1:])
        f1 = self.f1(individual)
        h = 1. - (f1 / g)**2
        return g*h

    def perfect_pareto_front(self):
        domain = seq(0, 1, 0.01)
        return domain, map(lambda x1: 1 - math.sqrt(x1), domain)
