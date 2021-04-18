###############################
#
# Created by Patrik Valkovic
# 4/15/2021
#
###############################
import os
import argparse
import ffeat
import ffeat.strategies as ES
import ffeat.measure as M
import bbobtorch
import torch as t
from utils import WandbExecutionTime, SidewayPipe, FSubtractPipe
import cpuinfo
import traceback

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = argparse.ArgumentParser()
p.add_argument('--dim', type=int, help="Problem dimension")
p.add_argument('--aarepeat', type=int, help="Repeat index")
p.add_argument('--iterations', type=int, help="number of iterations")
p.add_argument('--device', type=str, default='cpu', help="Where to run experiment")
p.add_argument('--popsize', type=int, help="Size of the population")
p.add_argument('--function', type=int, help="Function to evaluate")
p.add_argument('--selection', type=str, help="Selection to use")
p.add_argument('--crossover', type=str, help="Crossover to use")
p.add_argument('--crossover_offsprings', type=float, help="Number of offsprings for selection")
p.add_argument('--replace_parents', type=str, default='true', help="Whether should offsprings replace parents")
p.add_argument('--discard_parents', type=str, default='false', help="Whether should be parents discard after crossover")
p.add_argument('--crossover_params', type=str, help="Additional parameters for crossover")
p.add_argument('--mutation', type=str, help="Mutation to use")
p.add_argument('--mutation_params', type=str, help="Additional parameters for mutation")
args, unknown_args = p.parse_known_args()
args.repeat = args.aarepeat
if int(args.crossover_offsprings) == args.crossover_offsprings:
    args.crossover_offsprings = int(args.crossover_offsprings)
args.replace_parents = args.replace_parents.upper() == "TRUE"
args.discard_parents = args.discard_parents.upper() == "TRUE"
args.crossover_params = {e.split('-')[0]: float(e.split('-', maxsplit=1)[1]) for e in args.crossover_params.split(',')}
args.mutation_params = {e.split('-')[0]: float(e.split('-', maxsplit=1)[1]) for e in args.mutation_params.split(',')}
d = t.device(args.device)

fn = getattr(bbobtorch, f'create_f{args.function:02d}')(args.dim, dev=d)
selection = getattr(ES.selection, args.selection)(args.popsize)
crossover = getattr(ES.crossover, args.crossover)(
    args.crossover_offsprings,
    discard_parents=args.discard_parents,
    replace_parents=args.replace_parents,
    **args.crossover_params
)
mutation = getattr(ES.mutation, args.mutation)(
    **args.mutation_params
)
with WandbExecutionTime({'config': {
    'run_type': 'hyperparametersearch',
    'run_failed': False,
    'problem_group': 'bbob',
    'bbob_fn': args.function,
    'bbob_dim': args.dim,

    'alg_group': 'es',
    'pop_size': args.popsize,
    'es.selection': args.selection,
    'es.crossover': args.crossover,
    'es.crossover.offsprings': args.crossover_offsprings,
    'es.crossover.replace_parents': args.replace_parents,
    'es.crossover.discard_parents': args.discard_parents,
    **{f'es.crossover.params.{k}': v for k,v in args.crossover_params.items()},
    'es.mutation': args.mutation,
    **{f'es.mutation.params.{k}': v for k,v in args.mutation_params.items()},
    'es.elitism': False,

    'cputype': cpuinfo.get_cpu_info()['brand_raw'],
    'gputype': t.cuda.get_device_name(0) if t.cuda.is_available() else None,
}}) as reporter:
    try:
        alg = ES.EvolutionStrategy(
            ES.initialization.Uniform(args.popsize, -5, 5, args.dim, device=d),
            ES.evaluation.Evaluation(fn),
            SidewayPipe(
                FSubtractPipe(fn.f_opt),
                M.FitnessMean(),
                M.FitnessStd(),
                M.FitnessLowest(),
                M.Fitness01Quantile(),
                M.Fitness05Quantile(),
                M.FitnessMedian(),
                M.Fitness95Quantile(),
                M.Fitness99Quantile(),
                M.FitnessHighest(),
                reporter,
            ),
            selection,
            crossover,
            mutation,
            ffeat.pso.clip.Position(-5,5),
            iterations=args.iterations
        )
        alg(),
    except:
        traceback.print_exc()
        reporter.run.config.update({'run_failed': True}, allow_val_change=True)

