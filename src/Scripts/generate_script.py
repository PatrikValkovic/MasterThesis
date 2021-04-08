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
parser.add_argument('--output', type=str, help='Output directory')
args, _ = parser.parse_known_args()

for job_id, (fn, fdim) in enumerate(itertools.product(
    [1, 7, 15, 19, 22, 24],
    [6, 32, 128, 384]
)):
    with open(f'{args.output}/_job.{int(time.time())}.{job_id}.sh', "w") as f:
        print("#!/bin/bash")
        print("#PBS -N PSO_GPU")
        print("#PBS -q gpu")
        print("#PBS -l select=1:ncpus=4:mem=8gb:scratch_local=50gb:cluster=adan:ngpus=1")
        print("#PBS -l walltime=20:00:00")
        print()
        print("cp -r /storage/praha1/home/kowalsky/PYTHON \"$SCRATCHDIR\"")
        print("cp -r /storage/praha1/home/kowalsky/PIP \"$SCRATCHDIR\"")
        print("cp -r /storage/praha1/home/kowalsky/FFEAT \"$SCRATCHDIR\"")
        print("cp -r /storage/praha1/home/kowalsky/BBOBtorch \"$SCRATCHDIR\"")
        print()
        print("export PIP_CACHE_DIR=\"$SCRATCHDIR/pipcache\"")
        print("export PIP_TARGET=\"$SCRATCHDIR/PIP\"")
        print("export PYTHONPATH=\"$SCRATCHDIR/PIP\"")
        print("export PATH=\"$SCRATCHDIR/PYTHON:$PATH\"")
        print("export WANDB_API_KEY=\"799946cf5191b8c89fac5fcaafa346e1281ef80b\"")
        print("export WANDB_ENTITY=\"kowalsky\"")
        print("export WANDB_PROJECT=\"thesis\"")
        print("export TMP=\"$SCRATCHDIR/tmp\"")
        print("export TMPDIR=\"$SCRATCHDIR/tmp\"")
        print("export TEMP=\"$SCRATCHDIR/tmp\"")
        print()
        print("cd \"$SCRATCHDIR/BBOBtorch\"")
        print("python -m pip install --no-deps .")
        print("cd \"$SCRATCHDIR/FFEAT\"")
        print("python -m pip install --no-deps .")
        print()
        print("cp -r \"/storage/praha1/home/kowalsky/MasterThesis/src/Scripts\" \"$SCRATCHDIR\"")
        print("cd \"$SCRATCHDIR/Scripts\"")
        print(f"python fitness_pso2006_random.py --function {fn} --dim {fdim} --popsize \"32,512,10240,32768\" --device \"cuda\" --repeat 100 --cpu_count $(($PBS_NCPUS / 2))")
        print()
        print("rm -rf \"$SCRATCHDIR\"")
