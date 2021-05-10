###############################
#
# Created by Patrik Valkovic
# 5/7/2021
#
###############################
import ffeat.genetic as GA
import ffeat.measure as measure
from ffeat.utils import termination
from SATProblem import SATProblem

FILE = "uf250-017.cnf"
problem = SATProblem.from_cnf_file(FILE)

best_fitness = []
mean_fitness = []

alg = GA.GeneticAlgorithm(
    GA.initialization.Uniform(100, problem.nvars),
    GA.evaluation.Evaluation(problem.fitness_count_unsatisfied),
    measure.FitnessLowest(measure.reporting.Array(best_fitness)),  # report lowest fitness value
    measure.FitnessMean(measure.reporting.Array(mean_fitness)),  # report mean fitness value
    # terminate if algorithm found optimal solution
    termination.MetricReached(measure.FitnessLowest.ARG_NAME, 1, 0),
    # terminate if fitness mean does not improve for 100 steps
    termination.NoImprovement(measure.FitnessMean.ARG_NAME, 500),
    GA.selection.Tournament(100),  # Select 100 individuals
    GA.crossover.TwoPoint1D(0.6),  # Crossover 60% of them
    GA.mutation.FlipBit(0.4, 0.001),  # Mutate 40% of them with probability change 0.1%
    iterations=None,  # infinite iterations
)
alg()


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(best_fitness)), best_fitness, label='Best fitness')
plt.plot(range(len(mean_fitness)), mean_fitness, label='Mean fitness')
plt.legend()
plt.show()