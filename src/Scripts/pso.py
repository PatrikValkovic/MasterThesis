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


os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DIM = 80
POP_SIZE = 10000
ITERS = 10000
REPEAT = 10
d = t.device('cuda')


logs = []
def _save_log(*args, **kwargs):
    global logs
    logs.append(dict.copy(kwargs))
    return args, kwargs

for _ in range(REPEAT):
    fn = bbobtorch.create_f15(DIM, dev=d)
    with WandbReporter({'config': {
        'run_type': 'testing',
    }}) as reporter:
        try:
            alg = PSO.PSO(
                PSO.initialization.Uniform(POP_SIZE, -5, 5, DIM, device=d),
                PSO.initialization.Uniform(POP_SIZE, -1, 1, DIM, device=d),
                PSO.evaluation.Evaluation(fn),
                PSO.neighborhood.Random(10),
                PSO.update.PSO2006(0.8, 1.5, 1.5),
                measurements_termination=[
                    SidewayPipe(
                        FSubtractPipe(fn.f_opt),
                        M.FitnessMean(),
                        M.FitnessMedian(),
                        M.FitnessLowest(),
                        M.FitnessStd(),
                        M.Fitness05Quantile(),
                        reporter
                    )
                ],
                clip_velocity=PSO.clip.VelocityValue(-2, 2),
                clip_position=PSO.clip.Position(-5, 5),
                iterations=ITERS
            )
            alg()
        except:
            reporter.run.config.update({'run_failed': True})

#import matplotlib.pyplot as plt
#plt.figure(figsize=(8,6))
#plt.plot(range(len(logs)), list(map(lambda x: x[M.FitnessMean.ARG_NAME], logs)), c='tab:orange', linestyle='-', label='Mean')
#plt.plot(range(len(logs)), list(map(lambda x: x[M.FitnessStd.ARG_NAME], logs)), c='tab:orange', linestyle=':', label='STD')
#plt.plot(range(len(logs)), list(map(lambda x: x[M.FitnessMedian.ARG_NAME], logs)), c='tab:blue', linestyle='-', label="Median")
#plt.plot(range(len(logs)), list(map(lambda x: x[M.Fitness05Quantile.ARG_NAME], logs)), c='tab:blue', linestyle='--', label="Q 0.5")
#plt.plot(range(len(logs)), list(map(lambda x: x[M.FitnessLowest.ARG_NAME], logs)), c='tab:blue', linestyle=':', label="Minimum")
#plt.legend()
#plt.yscale('log')
#plt.show()
