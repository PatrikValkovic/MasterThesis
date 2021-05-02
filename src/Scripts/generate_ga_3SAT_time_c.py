###############################
#
# Created by Patrik Valkovic
# 4/19/2021
#
###############################
import argparse
import itertools
import time

from generate_ga_3SAT_fitness import popsize

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True, help='Output directory')
parser.add_argument("--cores", type=int, default=32, help="Number of real cores per machine")
parser.add_argument('--cores_per_job', type=int, default=8, help="Number of real cores per job")
parser.add_argument('--take_cores_per_job', type=int, default=16, help="Number of cores to ask per job")
args, _ = parser.parse_known_args()

jobs_per_machine = args.cores // args.cores_per_job
POP_SIZES = [
    '"32,1024,16384"',
    '"128,2048,32768"',
    '"200,5000"',
    '"512,10240"',
]
combinations = list(enumerate(itertools.product(
    POP_SIZES,
)))
ctime = int(time.time())
devices = ['pcr10']

for job_id, (popsize,) in combinations:
    with open(f'{args.output}/_job.{ctime}.{job_id}.sh', "w") as f:
        print(f"#!/bin/bash", file=f)
        print(f"#PBS -N GA_3SAT_C", file=f)
        print(f"#PBS -l select=1:ncpus={args.take_cores_per_job}:mem=32gb:scratch_local=50gb:vnode={devices[job_id % len(devices)]}", file=f)
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
        print("export REALJOBID=`echo $PBS_JOBID | cut -d'.' -f 1`", file=f)
        print(file=f)
        print("cd \"$SCRATCHDIR/BBOBtorch\"", file=f)
        print("python -m pip install --no-deps .", file=f)
        print("cd \"$SCRATCHDIR/FFEAT\"", file=f)
        print("python -m pip install --no-deps .", file=f)
        print(file=f)
        print("cp -r \"/storage/praha1/home/kowalsky/MasterThesis/src/Scripts\" \"$SCRATCHDIR\"", file=f)
        print("cd \"$SCRATCHDIR/Scripts\"", file=f)
        print(file=f)
        print(f"./ga.exe", end=" ", file=f)
        print(f"800", end=" ", file=f) # literals
        print(f"3600", end=" ", file=f) # clauses
        print(f"100", end=" ", file=f) # repeat
        print(f"1000", end=" ", file=f) # iterations
        print(f"{popsize}", end=" ", file=f) # popsize
        print(f"tmp.out", end=" ", file=f) # outfile
        print(f"8", end=" ", file=f) # threads
        print(f"100", end=" ", file=f) # min iterations
        print(f"60", end=" ", file=f) # max time
        print(f'cp tmp.out "$PBS_O_WORKDIR/$REALJOBID.out"', end=" ", file=f)
        print(file=f)
        print(file=f)
        print("rm -rf \"$SCRATCHDIR\"", file=f)
        print("echo \"DONE\"", file=f)
        combinations = combinations[jobs_per_machine:]
