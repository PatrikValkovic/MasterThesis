###############################
#
# Created by Patrik Valkovic
# 02.04.21
#
###############################
import ffeat.strategies as ES
from utils import MaxTimeMinItersTerminate as TimeTerminate, WandbExecutionTime
import bbobtorch

REPEAT = 10
DIM = 80
POP_SIZE = 1000
MAX_TIME = 5000
MIN_ITERS = 2000

for _ in range(REPEAT):
    fn = bbobtorch.create_f08(DIM)
    with WandbExecutionTime() as reporter:
        alg = ES.EvolutionStrategy(
            ES.initialization.Uniform(POP_SIZE, -5, 5, DIM),
            ES.evaluation.Evaluation(fn),
            TimeTerminate(MAX_TIME, MIN_ITERS),
            ES.selection.Tournament(1.0),
            ES.crossover.Blend(0.4, 0.5),
            ES.mutation.AddFromNormal(0.001, 0.3),
            reporter,
            iterations=None
        )
        r, k = alg()
