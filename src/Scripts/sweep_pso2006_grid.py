###############################
#
# Created by Patrik Valkovic
# 4/4/2021
#
###############################
import os
import argparse
import ffeat.pso as PSO
import ffeat.measure as M
import bbobtorch
import torch as t
from utils import WandbReporter, SidewayPipe, FSubtractPipe
import cpuinfo
import ffeat

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, help="Problem dimension")
p.add_argument('--aarepeat', type=int, help="Repeat index")
p.add_argument('--iterations', type=int, help="number of iterations")
p.add_argument("--device", type=str, default='cpu', help="Where to run experiment")
p.add_argument("--popsize", type=int, help="Size of the population")

p.add_argument("--inertia", type=float, help='Inertia weight')
p.add_argument("--local_c", type=float, help='Local c weight')
p.add_argument("--global_c", type=float, help='Global c weight')
p.add_argument("--velocity_clip", type=float, help="Where to clip velocity")
p.add_argument("--neigh_size", type=int, help="Neighbrhood size")
p.add_argument("--grid_type", type=str, help="Type of grid")
p.add_argument("--grid_size", type=str, help="Size of the grid")
args, _ = p.parse_known_args()
args.grid_size = tuple(map(int,args.grid_size.split(',')))
args.repeat = args.aarepeat

d = t.device(args.device)

fn = bbobtorch.create_f15(args.dim, dev=d)
with WandbReporter({'config': {
    'run_type': 'hyperparametersearch',
    'run_failed': False,
    'problem_group': 'bbob',
    'bbob_fn': 15,
    'bbob_dim': args.dim,
    'alg_group': 'pso',
    'pso.neigh': PSO.neighborhood.Grid2D.__name__,
    'pso.gridtype': args.grid_type,
    'pso.gridsize': args.grid_size,
    'pso.neigh.size': args.neigh_size,
    'pso.update': PSO.update.PSO2006.__name__,
    'pso.weight.inertia': args.inertia,
    'pso.local_c': args.local_c,
    'pso.global_c': args.global_c,
    'cputype': cpuinfo.get_cpu_info()['brand_raw'],
    'gputype': t.cuda.get_device_name(0) if t.cuda.is_available() else None,
}}) as reporter:
    try:
        alg = PSO.PSO(
            PSO.initialization.Uniform(args.popsize, -5, 5, args.dim, device=d),
            PSO.initialization.Uniform(args.popsize, -1, 1, args.dim, device=d),
            PSO.evaluation.Evaluation(fn),
            PSO.neighborhood.Grid2D(args.grid_type, args.neigh_size, shape=args.grid_size),
            PSO.update.PSO2006(args.inertia, args.local_c, args.global_c),
            measurements_termination=[
                SidewayPipe(
                    FSubtractPipe(fn.f_opt),
                    M.FitnessMean(),
                    M.FitnessMedian(),
                    M.FitnessLowest(),
                    M.FitnessStd(),
                    M.Fitness05Quantile(),
                    ffeat.utils.termination.NoImprovement(M.Fitness05Quantile.ARG_NAME,20),
                    reporter
                )
            ],
            clip_velocity=PSO.clip.VelocityValue(-args.velocity_clip, args.velocity_clip),
            clip_position=PSO.clip.Position(-5,5),
            iterations=args.iterations
        )
        alg()
    except:
        reporter.run.config.update({'run_failed': True})

