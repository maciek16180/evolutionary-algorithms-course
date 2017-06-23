"""Module with definition of MOTSP problem interface"""

from nsga2.individual import Individual
from nsga2.problems import Problem
import random, functools
import numpy as np

class MOTSP(Problem):

    def __init__(self, motsp_definitions):
        self.motsp_definitions = motsp_definitions
        self.max_objectives = [None, None]
        self.min_objectives = [None, None]
        self.problem_type = 'MOTSP'
        self.n = 100

    def __dominates(self, individual2, individual1):
        worse_than_other = self.motsp_definitions.f1(individual1) <= self.motsp_definitions.f1(individual2) and \
                           self.motsp_definitions.f2(individual1) <= self.motsp_definitions.f2(individual2)
        better_than_other = self.motsp_definitions.f1(individual1) < self.motsp_definitions.f1(individual2) or \
                            self.motsp_definitions.f2(individual1) < self.motsp_definitions.f2(individual2)
        return worse_than_other and better_than_other

    def generateIndividual(self, init=True):
        individual = Individual()
        if init:
            individual.features = list(np.random.permutation(self.n))
            self.calculate_objectives(individual)
        individual.dominates = functools.partial(self.__dominates, individual1=individual)
        return individual

    def calculate_objectives(self, individual):
        individual.objectives = []
        individual.objectives.append(self.motsp_definitions.f1(individual))
        individual.objectives.append(self.motsp_definitions.f2(individual))
        for i in range(2):
            if self.min_objectives[i] is None or individual.objectives[i] < self.min_objectives[i]:
                self.min_objectives[i] = individual.objectives[i]
            if self.max_objectives[i] is None or individual.objectives[i] > self.max_objectives[i]:
                self.max_objectives[i] = individual.objectives[i]
