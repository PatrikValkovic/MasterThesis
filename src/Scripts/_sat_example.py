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

generate_cnf_norm(200, 1200, 0, 3, 0, FILEPATH)
s = SAT.from_cnf_file(FILEPATH)

def _log_iteration(*_, **__):
    print(f"Iteration {__['iteration']}")
    return _, __

start_time = time.time()
alg = GA.GeneticAlgorithm(
    GA.initialization.Uniform(100, s.nvars, dtype=t.int8, device=d),
    GA.evaluation.Evaluation(s.fitness_count_unsatisfied),
    _log_iteration,
    M.FitnessMedian(M.reporting.Console('median')),
    M.FitnessMean(M.reporting.Console('mean')),
    M.FitnessLowest(M.reporting.Console('lowest')),
    GA.selection.Tournament(100),
    GA.crossover.TwoPoint1D(0.4),
    GA.mutation.FlipBit(0.6, mutate_prob=0.001),
    iterations=10000
)
alg()
print(f"TOOK {time.time() - start_time}")

os.remove(FILEPATH)
