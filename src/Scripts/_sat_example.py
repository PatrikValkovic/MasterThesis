###############################
#
# Created by Patrik Valkovic
# 03.04.21
#
###############################
import ffeat.genetic as GA
import ffeat.measure as M
from problems import SAT
import torch as t
import time
from utils import generate_cnf_norm
import os

FILEPATH = 'tmp.cnf'
cpu = t.device('cpu')
gpu = t.device('cuda')
d = cpu
t.rand(1,device=d)

generate_cnf_norm(200, 2000, 0, 3, 0, FILEPATH)
s = SAT.from_cnf_file(FILEPATH)

mean_f, median_f, min_f = [], [], []
alg = GA.GeneticAlgorithm(
    GA.initialization.Uniform(100, s.nvars, dtype=t.int8, device=d),
    GA.evaluation.Evaluation(s.fitness_count_unsatisfied),
    M.FitnessMedian(M.reporting.Array(median_f)),
    M.FitnessMean(M.reporting.Array(mean_f)),
    M.FitnessLowest(M.reporting.Array(min_f)),
    GA.selection.Tournament(100),
    GA.crossover.TwoPoint1D(0.4),
    GA.mutation.FlipBit(0.6, mutate_prob=0.0001),
    iterations=1000
)
alg()

os.remove(FILEPATH)

import matplotlib.pyplot as plt
plt.plot(range(len(mean_f)), mean_f, label='Mean')
plt.plot(range(len(median_f)), median_f, label='Median')
plt.plot(range(len(min_f)), min_f, label='Min')
plt.legend()
plt.show()
