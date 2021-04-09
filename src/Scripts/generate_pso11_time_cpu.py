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
parser.add_argument('--cores_per_job', default=8, help="Number of real cores per job")
args, _ = parser.parse_known_args()

jobs_per_machine = args.cores // args.cores_per_job
combinations = list(enumerate(itertools.product(
    [6, 32, 128, 384],
    [1, 7, 15, 19, 22, 24],
)))

for job_id, (fdim, fn) in combinations:
    with open(f'{args.output}/_job.{int(time.time())}.{job_id}.sh', "w") as f:
        print("#!/bin/bash", file=f)
        print("#PBS -N PSO_CPU_time", file=f)
        print("#PBS -l select=1:ncpus:16:mem=32gb:scratch_local=50gb:cluster=pcr", file=f)
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
        print(f"python time_pso2011_random.py --function {fn} --dim {fdim} --popsize \"32,128,200,512,1024,2048,5000,10240,16384,32768\" --device \"cpu\" --repeat 100 --cpu_count {args.cores_per_job} &", file=f)
        print("wait", file=f)
        print(file=f)
        print("rm -rf \"$SCRATCHDIR\"", file=f)
        combinations = combinations[jobs_per_machine:]
