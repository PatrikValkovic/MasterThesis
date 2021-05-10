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

POP_SIZE = 1000
LITERALS = 2000
CLAUSES = [
    '"32,2048,32768"',
    '"128,1024,5000"',
    '"200,10240"',
    '"512,16384"',
]
combinations = list(enumerate(itertools.product(
    CLAUSES
)))
ctime = int(time.time())

for job_id, (clauses,) in combinations:
    with open(f'{args.output}/_job.{ctime}.{job_id}.sh', "w") as f:
        print(f"#!/bin/bash", file=f)
        print(f"#PBS -N GPU_GA_3SAT_clausecount", file=f)
        print(f"#PBS -q gpu", file=f)
        print(f"#PBS -l select=1:ncpus=4:mem=8gb:scratch_local=50gb:cluster=adan:ngpus=1", file=f)
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
        print(f"python ./time_ga_3SAT_clausecount.py", end=" ", file=f)
        print(f"--repeat 100 --iterations 1000", end=" ", file=f)
        print(f"--device cuda --cpu_count $(($PBS_NCPUS / 2))", end=" ", file=f)
        print(f"--popsize {POP_SIZE}", end=" ", file=f)
        print(f"--literals {LITERALS}", end=" ", file=f)
        print(f"--clauses {clauses}", end=" ", file=f)
        print(f"--mean_literals_in_clause 3 --std_literals_in_clause 0", end=" ", file=f)
        print(f"2>&1", end=" ", file=f)
        print(file=f)
        print(file=f)
        print("rm -rf \"$SCRATCHDIR\"", file=f)
        print("echo \"DONE\"", file=f)
