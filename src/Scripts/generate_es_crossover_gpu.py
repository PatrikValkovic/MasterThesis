###############################
#
# Created by Patrik Valkovic
# 4/19/2021
#
###############################
import argparse
import itertools
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True, help='Output directory')
parser.add_argument("--cores", type=int, default=32, help="Number of real cores per machine")
parser.add_argument('--cores_per_job', type=int, default=8, help="Number of real cores per job")
parser.add_argument('--take_cores_per_job', type=int, default=16, help="Number of cores to ask per job")
args, _ = parser.parse_known_args()

jobs_per_machine = args.cores // args.cores_per_job
CROSSOVERS = [
    {'crossover': 'Arithmetic', 'offsprings': 0.4, 'params': None},
    {'crossover': 'Blend', 'offsprings': 0.7, 'params': 'alpha-0.5'},
    {'crossover': 'OnePoint1D', 'offsprings': 0.8, 'params': None},
    {'crossover': 'TwoPoint1D', 'offsprings': 0.8, 'params': None},
    {'crossover': 'Uniform', 'offsprings': 0.8, 'params': 'change_prob-0.4'},
]
POP_SIZES = [
    '"2048,32,128,200,512,1024,5000,10240,16384"',
    '32768'
]
TO_MEASURE = '"32,512,10240,32768"'
combinations = list(enumerate(itertools.product(
    [24, 128, 384],  # fdim
    [19, 24],  # fn
    CROSSOVERS,
    POP_SIZES,
)))
ctime = int(time.time())

for job_id, (fdim, fn, cross, popsize) in combinations:
    with open(f'{args.output}/_job.{ctime}.{job_id}.sh', "w") as f:
        print(f"#!/bin/bash", file=f)
        print(f"#PBS -N GPU_ES_CROSSOVER", file=f)
        print("#PBS -q gpu", file=f)
        print("#PBS -l select=1:ncpus=4:mem=8gb:scratch_local=50gb:cluster=adan:ngpus=1", file=f)
        print(f"#PBS -l walltime=24:00:00", file=f)
        print(file=f)
        print("cp -r /storage/praha1/home/kowalsky/PYTHON \"$SCRATCHDIR\"", file=f)
        print("cp -r /storage/praha1/home/kowalsky/PIP \"$SCRATCHDIR\"", file=f)
        print("cp -r /storage/praha1/home/kowalsky/FFEAT \"$SCRATCHDIR\"", file=f)
        print("cp -r /storage/praha1/home/kowalsky/BBOBtorch \"$SCRATCHDIR\"", file=f)
        print(file=f)
        print("export PIP_CACHE_DIR=\"$SCRATCHDIR/pipcache\"", file=f)
        print("export PIP_TARGET=\"$SCRATCHDIR/PIP\"", file=f)
        print("export PYTHONPATH=\"$SCRATCHDIR/PIP\"", file=f)
        print("export PATH=\"$SCRATCHDIR/PYTHON:$PATH\"", file=f)
        print("export WANDB_API_KEY=\"799946cf5191b8c89fac5fcaafa346e1281ef80b\"", file=f)
        print("export WANDB_ENTITY=\"kowalsky\"", file=f)
        print("export WANDB_PROJECT=\"thesis\"", file=f)
        print("export TMP=\"$SCRATCHDIR/tmp\"", file=f)
        print("export TMPDIR=\"$SCRATCHDIR/tmp\"", file=f)
        print("export TEMP=\"$SCRATCHDIR/tmp\"", file=f)
        print(file=f)
        print("cd \"$SCRATCHDIR/BBOBtorch\"", file=f)
        print("python -m pip install --no-deps .", file=f)
        print("cd \"$SCRATCHDIR/FFEAT\"", file=f)
        print("python -m pip install --no-deps .", file=f)
        print(file=f)
        print("cp -r \"/storage/praha1/home/kowalsky/MasterThesis/src/Scripts\" \"$SCRATCHDIR\"", file=f)
        print("cd \"$SCRATCHDIR/Scripts\"", file=f)
        print(file=f)
        print(f"python ./timefitness_es_crossover.py", end=" ", file=f)
        print(f"--mutation AddFromNormal --mutation_params \"mutation_rate-0.2,std-0.005\"", end=" ", file=f)
        print(f"--function {fn} --dim {fdim}", end=" ", file=f)
        print(f"--repeat 100 --iterations 1000", end=" ", file=f)
        print(f"--device cuda --cpu_count $(($PBS_NCPUS / 2))", end=" ", file=f)
        print(f"--selection Tournament", end=" ", file=f)
        print(f"--crossover {cross['crossover']}", end=" ", file=f)
        print(f"--crossover_offsprings {cross['offsprings']}", end=" ", file=f)
        if cross['params'] is not None:
            print(f"--crossover_params {cross['params']}", end=" ", file=f)
        print(f"--replace_parents true --discard_parents false", end=" ", file=f)
        print(f"--popsize {popsize}", end=" ", file=f)
        print(f"--measure {TO_MEASURE}", end=" ", file=f)
        print(file=f)
        print(file=f)
        print("rm -rf \"$SCRATCHDIR\"", file=f)
        print("echo \"DONE\"", file=f)
        combinations = combinations[jobs_per_machine:]
