###############################
#
# Created by Patrik Valkovic
# 4/19/2021
#
###############################
import os
import time
import argparse
import ffeat.measure as M
import torch as t
from utils import WandbExecutionTime, MaxTimeMinItersTerminate, FSubtractFromPipe, generate_cnf_norm, SidewayPipe
import cpuinfo
import itertools
import ffeat.genetic as GA
import traceback
from problems import SAT


os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = argparse.ArgumentParser()
p.add_argument('--repeat', type=int, default=100, help="Repeat index")
p.add_argument('--iterations', type=int, default=1000, help="number of iterations")
p.add_argument("--device", type=str, default='cuda', help="Where to run experiment")
p.add_argument("--cpu_count", type=int, default=None, help="Number of GPU to use")

p.add_argument("--popsize", type=str, help="Size of the population comma separated")
p.add_argument("--literals", type=str, help="Number of literals")
p.add_argument('--mean_literals_in_clause', type=float, help="Expected number of literals in clause")
p.add_argument('--std_literals_in_clause', type=float, help="Deviation of number of literals in clause")
p.add_argument('--measure', type=str, help="Which population size to measure")
args, _ = p.parse_known_args()
args.popsize = list(map(int, args.popsize.split(',')))
args.literals = list(map(int, args.literals.split(',')))
args.measure = list(map(int, args.measure.split(',')))

dev = t.device(args.device)
print(f'GONNA USE {dev}')
if args.cpu_count is not None:
    t.set_num_threads(args.cpu_count)
float(t.rand(10).max())

crossover = GA.crossover.OnePoint1D(0.4)
mutation = GA.mutation.FlipBit(0.6, 0.001)
for psize, literals in itertools.product(args.popsize, args.literals):
    clauses = int(literals * 4.5)
    selection = GA.selection.Tournament(psize)
    for i in range(args.repeat):
        generate_cnf_norm(literals, clauses, 0, args.mean_literals_in_clause, args.std_literals_in_clause, 'tmp.cnf')
        fn = SAT.from_cnf_file("tmp.cnf", dev)
        print(f"{time.asctime()}\tSAT {literals}:{clauses} with population {psize}, running for {i}")
        with WandbExecutionTime({'config': {
            **vars(args),
            'run_type': 'time,fitness',
            'run_failed': False,
            'problem_group': 'sat',
            'sat.literals': literals,
            'sat.clauses': clauses,
            'sat.mliter': args.mean_literals_in_clause,
            'sat.sliter': args.std_literals_in_clause,

            'alg_group': 'ga_1',
            'pop_size': psize,
            'ga.selection': selection.__class__.__name__,
            'ga.crossover': crossover.__class__.__name__,
            'ga.crossover.offsprings': 0.4,
            'ga.crossover.replace_parents': 'true',
            'ga.crossover.discard_parents': 'false',
            'ga.mutation': GA.mutation.FlipBit.__name__,
            'ga.mutation.params.to_mutate': 0.6,
            'ga.mutation.params.mutation_prob': 0.001,
            'ga.elitism': False,

            'cputype': cpuinfo.get_cpu_info()['brand_raw'],
            'gputype': t.cuda.get_device_name(0) if t.cuda.is_available() else None,
        }}) as reporter:
            try:
                alg = GA.GeneticAlgorithm(
                    GA.initialization.Uniform(psize, literals, device=dev),
                    GA.evaluation.Evaluation(fn.fitness_count_unsatisfied),
                    SidewayPipe(
                        FSubtractFromPipe(clauses),
                        M.FitnessMean(),
                        M.FitnessStd(),
                        M.FitnessLowest(),
                        M.Fitness01Quantile(),
                        M.Fitness05Quantile(),
                        M.FitnessMedian(),
                        reporter,
                        MaxTimeMinItersTerminate(1 * 60 * 1000, 80 if (psize not in args.measure) else args.iterations),
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