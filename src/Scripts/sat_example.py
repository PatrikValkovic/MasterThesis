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


FILEPATH = '/home/kowalsky/Downloads/eva/uf20-01.cnf'


s = SAT.from_cnf_file(FILEPATH)

alg = GA.GeneticAlgorithm(
    GA.initialization.Uniform(100, s.nvars, dtype=t.int8),
    GA.evaluation.Evaluation(s.unoptimized),
    M.FitnessMedian(M.reporting.Console('median')),
    M.FitnessMean(M.reporting.Console('mean')),
    M.FitnessLowest(M.reporting.Console('lowest')),
    GA.selection.Tournament(1.0),
    GA.crossover.TwoPoint1D(0.4, replace_parents=False),
    GA.mutation.FlipBit(0.1, mutate_prob=0.01),
    iterations=100
)
alg()
