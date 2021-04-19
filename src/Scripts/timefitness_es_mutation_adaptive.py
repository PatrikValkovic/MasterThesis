###############################
#
# Created by Patrik Valkovic
# 4/19/2021
#
###############################
import os
import sys
import argparse
import bbobtorch
import torch as t
from utils import WandbExecutionTime, SidewayPipe, FSubtractPipe
import ffeat.measure as M
import cpuinfo
import itertools
import ffeat.strategies as ES
import ffeat
import traceback

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = argparse.ArgumentParser()
p.add_argument('--repeat', type=int, default=100, help="Repeat index")
p.add_argument('--iterations', type=int, default=1000, help="number of iterations")
p.add_argument("--device", type=str, default='cpu', help="Where to run experiment")
p.add_argument("--cpu_count", type=int, default=None, help="Number of GPU to use")

p.add_argument('--selection', type=str, help="Selection to use")
p.add_argument('--crossover', type=str, help="Crossover to use")
p.add_argument('--crossover_offsprings', type=float, help="Number of offsprings for selection")
p.add_argument('--replace_parents', type=str, default='true', help="Whether should offsprings replace parents")
p.add_argument('--discard_parents', type=str, default='false', help="Whether should be parents discard after crossover")
p.add_argument('--crossover_params', type=str, help="Additional parameters for crossover")

p.add_argument("--function", type=str, help="BBOB function index")
p.add_argument("--dim", type=str, help="Size of the problem comma separated")
p.add_argument("--popsize", type=str, help="Size of the population comma separated")
args, _ = p.parse_known_args()
args.mutation_params = 'init_std-0.01,std_increase-1.3,std_decrease-0.2,mutate_members-1.0,better_to_increase-0.3,minimum_std-0.00001,maximum_std-1'

args.function = list(map(int, args.function.split(',')))
args.dim = list(map(int, args.dim.split(',')))
args.popsize = list(map(int, args.popsize.split(',')))
args.replace_parents = args.replace_parents.upper() == "TRUE"
args.discard_parents = args.discard_parents.upper() == "TRUE"
args.crossover_params = {e.split('-')[0]: float(e.split('-', maxsplit=1)[1]) for e in args.crossover_params.split(',')}
args.mutation_params = {e.split('-')[0]: float(e.split('-', maxsplit=1)[1]) for e in args.mutation_params.split(',')}

dev = t.device(args.device)
print(f'GONNA USE {dev}')
if args.cpu_count is not None:
    t.set_num_threads(args.cpu_count)
float(t.rand(10).max())

crossover = getattr(ES.crossover, args.crossover)(
        args.crossover_offsprings,
        discard_parents=args.discard_parents,
        replace_parents=args.replace_parents,
        **args.crossover_params
    )

for fni, d, psize in itertools.product(args.function, args.dim, args.popsize):
    fn = getattr(bbobtorch, f"create_f{fni:02d}")(d, dev=dev)
    selection = getattr(ES.selection, args.selection)(psize)
    mutation = ES.mutation.AdaptiveStep(evaluation=ES.evaluation.Evaluation(fn), **args.mutation_params)

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
            'alg_group': 'es_mutation',
            'es.selection': args.selection,
            'es.crossover': args.crossover,
            'es.crossover.offsprings': args.crossover_offsprings,
            'es.crossover.replace_parents': args.replace_parents,
            'es.crossover.discard_parents': args.discard_parents,
            **{f'es.crossover.params.{k}': v for k,v in args.crossover_params.items()},
            'es.mutation': ES.mutation.AdaptiveStep.__name__,
            **{f'es.mutation.params.{k}': v for k,v in args.mutation_params.items()},
            'es.elitism': False,
        }}) as reporter:
            try:
                alg = ES.EvolutionStrategy(
                    ES.initialization.Uniform(psize, -5, 5, d, device=dev),
                    ES.evaluation.Evaluation(fn),
                    SidewayPipe(
                        FSubtractPipe(fn.f_opt),
                        M.FitnessMean(),
                        M.FitnessStd(),
                        M.FitnessLowest(),
                        M.Fitness01Quantile(),
                        M.Fitness05Quantile(),
                        M.FitnessMedian(),
                        reporter,
                    ),
                    mutation,
                    selection,
                    crossover,
                    ffeat.pso.clip.Position(-5, 5),
                    iterations=args.iterations
                )
                alg()
            except:
                traceback.print_exc()
                reporter.run.config.update({'run_failed': True}, allow_val_change=True)