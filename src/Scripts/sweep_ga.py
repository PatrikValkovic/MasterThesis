###############################
#
# Created by Patrik Valkovic
# 4/15/2021
#
###############################
import os
import argparse
import ffeat
import ffeat.genetic as GA
import ffeat.measure as M
import torch as t
from utils import WandbExecutionTime, SidewayPipe, FSubtractFromPipe, generate_cnf_norm
from problems import SAT
import cpuinfo
import traceback

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = argparse.ArgumentParser()
p.add_argument('--aarepeat', type=int, help="Repeat index")
p.add_argument('--popsize', type=int, help="Size of the population")
p.add_argument('--device', type=str, help="Where to run experiment")
p.add_argument('--literals', type=int, help="Number of literals")
p.add_argument('--clauses', type=int, help="Number of clauses")
p.add_argument('--mean_literals_in_clause', type=float, help="Expected number of literals in clause")
p.add_argument('--std_literals_in_clause', type=float, help="Deviation of number of literals in clause")
p.add_argument('--file', type=str, default='tmp.cnf', help="Where to store the CNF file")
p.add_argument('--iterations', type=int, help="number of iterations")
p.add_argument('--to_mutate', type=float, help="Number of individuals to mutate")
p.add_argument('--mutation_prob', type=float, help="Probability of mutation")
p.add_argument('--selection', type=str, help="Selection to use")
p.add_argument('--crossover', type=str, help="Crossover to use")
p.add_argument('--crossover_offsprings', type=float, help="Number of offsprings for selection")
p.add_argument('--replace_parents', type=str, default='true', help="Whether should offsprings replace parents")
p.add_argument('--discard_parents', type=str, default='false', help="Whether should be parents discard after crossover")
p.add_argument('--crossover_params', type=str, help="Additional parameters for crossover")
args, unknown_args = p.parse_known_args()
args.repeat = args.aarepeat
if int(args.crossover_offsprings) == args.crossover_offsprings:
    args.crossover_offsprings = int(args.crossover_offsprings)
args.replace_parents = args.replace_parents.upper() == "TRUE"
args.discard_parents = args.discard_parents.upper() == "TRUE"
args.crossover_params = {e.split('-')[0]: float(e.split('-', maxsplit=1)[1]) for e in args.crossover_params.split(',')} if args.crossover_params is not None else dict()
d = t.device(args.device)

generate_cnf_norm(args.literals, args.clauses, 0, args.mean_literals_in_clause, args.std_literals_in_clause, args.file)
problem_fn = SAT.from_cnf_file(args.file, d)
selection = getattr(GA.selection, args.selection)(args.popsize, **({'maximization': True} if args.selection == GA.selection.Tournament.__name__ else {}))
crossover = getattr(GA.crossover, args.crossover)(
    args.crossover_offsprings,
    discard_parents=args.discard_parents,
    replace_parents=args.replace_parents,
    **args.crossover_params
)
mutation = GA.mutation.FlipBit(args.to_mutate, args.mutation_prob)
with WandbExecutionTime({'config': {
    'run_type': 'hyperparametersearch',
    'run_failed': False,
    'problem_group': 'sat',
    'sat.literals': args.literals,
    'sat.clauses': args.clauses,
    'sat.mliter': args.mean_literals_in_clause,
    'sat.sliter': args.std_literals_in_clause,

    'alg_group': 'ga',
    'pop_size': args.popsize,
    'ga.selection': args.selection,
    'ga.crossover': args.crossover,
    'ga.crossover.offsprings': args.crossover_offsprings,
    'ga.crossover.replace_parents': args.replace_parents,
    'ga.crossover.discard_parents': args.discard_parents,
    **{f'ga.crossover.params.{k}': v for k,v in args.crossover_params.items()},
    'ga.mutation': GA.mutation.FlipBit.__name__,
    'ga.mutation.params.to_mutate': args.to_mutate,
    'ga.mutation.params.mutation_prob': args.mutation_prob,
    'ga.elitism': False,

    'cputype': cpuinfo.get_cpu_info()['brand_raw'],
    'gputype': t.cuda.get_device_name(0) if t.cuda.is_available() else None,
}}) as reporter:
    try:
        alg = GA.GeneticAlgorithm(
            GA.initialization.Uniform(args.popsize, args.literals, device=d),
            GA.evaluation.Evaluation(problem_fn.fitness_count_satisfied),
            SidewayPipe(
                FSubtractFromPipe(args.clauses),
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
            iterations=args.iterations
        )
        alg()
    except:
        traceback.print_exc()
        reporter.run.config.update({'run_failed': True}, allow_val_change=True)
