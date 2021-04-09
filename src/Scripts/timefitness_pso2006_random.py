###############################
#
# Created by Patrik Valkovic
# 4/4/2021
#
###############################
import os
import sys
import argparse
import ffeat.pso as PSO
import ffeat.measure as M
import bbobtorch
import torch as t
from utils import WandbExecutionTime, SidewayPipe, FSubtractPipe
import cpuinfo
import itertools

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = argparse.ArgumentParser()
p.add_argument('--repeat', type=int, default=100, help="Repeat index")
p.add_argument('--iterations', type=int, default=1000, help="number of iterations")
p.add_argument("--device", type=str, default='cpu', help="Where to run experiment")
p.add_argument("--velocity_clip", type=float, default=2, help="Velocity clip")
p.add_argument("--inertia", type=float, default=0.8, help="Weight inertia")
p.add_argument("--local_c", type=float, default=1.5)
p.add_argument("--global_c", type=float, default=1.5)
p.add_argument("--neigh_size", type=float, default=0.05, help="Neighborhood size")
p.add_argument("--cpu_count", type=int, default=None, help="Number of GPU to use")

p.add_argument("--function", type=str, help="BBOB function index")
p.add_argument("--dim", type=str, help="Size of the problem comma separated")
p.add_argument("--popsize", type=str, help="Size of the population comma separated")
args, _ = p.parse_known_args()
args.function = list(map(int, args.function.split(',')))
args.dim = list(map(int, args.dim.split(',')))
args.popsize = list(map(int, args.popsize.split(',')))

dev = t.device(args.device)
print(f'GONNA USE {dev}')
if args.cpu_count is not None:
    t.set_num_threads(args.cpu_count)

for fni, d, psize in itertools.product(args.function, args.dim, args.popsize):
    fn = getattr(bbobtorch, f"create_f{fni:02d}")(d, dev=dev)
    for i in range(args.repeat):
        with WandbExecutionTime({'config': {
            **vars(args),
            'run_type': 'time,fitness',
            'run_failed': False,
            'cputype': cpuinfo.get_cpu_info()['brand_raw'],
            'gputype': t.cuda.get_device_name(0) if t.cuda.is_available() else None,
            'device': args.device,
            'repeat': i,
            'pop_size': psize,
            'problem_group': 'bbob',
            'bbob_fn': fni,
            'bbob_dim': d,
            'alg_group': 'pso',
            'pso.neigh': PSO.neighborhood.Random.__name__,
            'pso.neigh.size': args.neigh_size,
            'pso.update': PSO.update.PSO2006.__name__,
            'pso.inertia': args.inertia,
            'pso.local_c': args.local_c,
            'pso.global_c': args.global_c,
        }}) as reporter:
            try:
                alg = PSO.PSO(
                    PSO.initialization.Uniform(psize, -5, 5, d, device=dev),
                    PSO.initialization.Uniform(psize, -1, 1, d, device=dev),
                    PSO.evaluation.Evaluation(fn),
                    PSO.neighborhood.Random(args.neigh_size),
                    PSO.update.PSO2006(args.inertia, args.local_c, args.global_c),
                    measurements_termination=[
                        SidewayPipe(
                            FSubtractPipe(fn.f_opt),
                            M.FitnessMean(),
                            M.FitnessStd(),
                            M.FitnessLowest(),
                            M.Fitness01Quantile(),
                            M.Fitness05Quantile(),
                            M.FitnessMedian(),
                            reporter,
                        )
                    ],
                    clip_velocity=PSO.clip.VelocityValue(-args.velocity_clip, args.velocity_clip),
                    clip_position=PSO.clip.Position(-5,5),
                    iterations=args.iterations
                )
                alg()
            except:
                print("Unexpected error:", sys.exc_info()[0])
                reporter.run.config.update({'run_failed': True}, allow_val_change=True)

