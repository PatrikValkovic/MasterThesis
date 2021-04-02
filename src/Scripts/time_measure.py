###############################
#
# Created by Patrik Valkovic
# 02.04.21
#
###############################
import ffeat.measure as M
import ffeat.strategies as ES
from MaxTimeMinItersTerminate import MaxTimeMinItersTerminate as TimeTerminate
from WandbReporter import WandbReporter
import bbobtorch

REPEAT = 10
DIM = 80
POP_SIZE = 1000
MAX_TIME = 5000
MIN_ITERS = 2000

for _ in range(REPEAT):
    fn = bbobtorch.create_f08(DIM)
    with WandbReporter() as reporter:
        alg = ES.EvolutionStrategy(
            ES.initialization.Uniform(POP_SIZE, -5, 5, DIM),
            ES.evaluation.Evaluation(fn),
            TimeTerminate(MAX_TIME, MIN_ITERS),
            M.FitnessLowest(),
            M.FitnessHighest(),
            M.FitnessMean(),
            M.FitnessMedian(),
            M.FitnessStd(),
            reporter,
            ES.selection.Tournament(1.0),
            ES.crossover.Blend(0.4, 0.5),
            ES.mutation.AddFromNormal(0.001, 0.3),
            iterations=None
        )
        r, k = alg()
