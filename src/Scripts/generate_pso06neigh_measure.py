###############################
#
# Created by Patrik Valkovic
# 08.04.21
#
###############################
import argparse
import itertools
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True, help='Output directory')
parser.add_argument('--cluster', type=str, default='pcr', help="Cluster where to run script")
parser.add_argument("--cores", type=int, default=32, help="Number of real cores per machine")
parser.add_argument('--cores_per_job', type=int, default=8, help="Number of real cores per job")
parser.add_argument('--take_cores_per_job', type=int, default=16, help="Number of cores to ask per job")
args, _ = parser.parse_known_args()

jobs_per_machine = args.cores // args.cores_per_job
combinations = list(enumerate(itertools.product(
    ['circle','nearest','grid_linear','grid_compact','grid_diamond'],
    [1, 7, 15, 19, 22, 24],
)))
ctime = int(time.time())
hyper = {
    'circle': 0.13,
    'nearest': 0.13,
    'grid_linear': 0.005,
    'grid_compact': 0.003,
    'grid_diamond': 0.005,
}

for job_id, (neigh_type, fn) in combinations:
    with open(f'{args.output}/_job.{ctime}.{job_id}.sh', "w") as f:
        print(f"#!/bin/bash", file=f)
        print("#PBS -N PSO06_GPU_neigh", file=f)
        print("#PBS -q gpu", file=f)
        print("#PBS -l select=1:ncpus=4:mem=8gb:scratch_local=50gb:cluster=adan:ngpus=1", file=f)
        print("#PBS -l walltime=24:00:00", file=f)
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
        print(f"python timefitness_pso2006_neighborhood.py --neigh_size {hyper[neigh_type]} --neigh_type {neigh_type} --function {fn} --dim 128 --popsize \"1225,36,121,225,529,2500,4900,10000,16900,22500\" --device \"cuda\" --repeat 100 --cpu_count $(($PBS_NCPUS / 2))", file=f)
        print(file=f)
        print("rm -rf \"$SCRATCHDIR\"", file=f)
        combinations = combinations[jobs_per_machine:]
