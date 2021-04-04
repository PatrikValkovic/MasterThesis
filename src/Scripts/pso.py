###############################
#
# Created by Patrik Valkovic
# 4/4/2021
#
###############################
import os
import torch as t
import bbobtorch
import ffeat.pso as PSO
import ffeat.measure as M
from utils import WandbReporter, SidewayPipe, FSubtractPipe
from progressbar import progressbar

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

REPEAT = 10
DIM = 80
POP_SIZE = 100
ITERS = 100
d = t.device('cpu')


for _ in progressbar(range(REPEAT)):
    fn = bbobtorch.create_f16(DIM, dev=d)
    with WandbReporter({'config': {
        'run_type': 'test',
        'run_failed': False,
        'problem_group': 'bbob',
        'bbob_fn': 16,
        'bbob_dim': DIM,
        'alg_group': 'pso',
        'pso.neigh': PSO.neighborhood.Random.__name__,
        'pso.neigh.size': 4,
        'pso.update': PSO.update.PSO2006.__name__,
        'pso.weight.inertia': 0.7,
        'pso.local_c': 1.2,
        'pso.global_c': 1.2,
    }}) as reporter:
        try:
            alg = PSO.PSO(
                PSO.initialization.Uniform(POP_SIZE, -5, 5, DIM, device=d),
                PSO.initialization.Uniform(POP_SIZE, -1, 1, DIM, device=d),
                PSO.evaluation.Evaluation(fn),
                PSO.neighborhood.Random(4),
                PSO.update.PSO2006(0.7, 1.2, 1.2),
                measurements_termination=[
                    SidewayPipe(
                        FSubtractPipe(fn.f_opt),
                        M.FitnessMean(),
                        M.FitnessMedian(),
                        M.FitnessLowest(),
                        reporter
                    )
                ],
                clip_velocity=PSO.clip.VelocityValue(-2, 2),
                clip_position=PSO.clip.Position(-5,5),
                iterations=ITERS
            )
            alg()
        except:
            reporter.run.config.update({'run_failed': True})
