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
args, _ = parser.parse_known_args()

MUTATIONS = [
    {
        'script': 'timefitness_es_mutation.py',
        'mutation': 'AddFromNormal',
        'mutation_params': 'mutation_rate-0.2,std-0.005',
    },
    {
        'script': 'timefitness_es_mutation.py',
        'mutation': 'AddFromCauchy',
        'mutation_params': 'mutation_rate-0.2,scale-0.003',
    },
    {
        'script': 'timefitness_es_mutation.py',
        'mutation': 'ReplaceUniform',
        'mutation_params': 'min--5,max-5,mutation_rate-0.2',
    },
    {
        'script': 'timefitness_es_mutation_adaptive.py',
        'mutation': 'AdaptiveStep',
        'mutation_params': 'nothing',
    },
]
combinations = list(enumerate(itertools.product(
    [24, 128, 384],  # fdim
    [19, 22],  # fn
    MUTATIONS,
)))
ctime = int(time.time())
devices = ['pcr16', 'pcr15']

for job_id, (fdim, fn, mut) in combinations:
    with open(f'{args.output}/_job.{ctime}.{job_id}.sh', "w") as f:
        print(f"#!/bin/bash", file=f)
        print(f"#PBS -N GPU_ES_MUTATION", file=f)
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
        print(f"python ./{mut['script']} --mutation \"{mut['mutation']}\" --mutation_params \"{mut['mutation_params']}\" --function {fn} --dim {fdim} --repeat 100 --iterations 1000 --device cuda --cpu_count $(($PBS_NCPUS / 2)) --selection Tournament --crossover Uniform --crossover_offsprings 0.8 --replace_parents true --discard_parents false --crossover_params \"change_prob-0.4\" --popsize \"2048,32,128,200,512,1024,5000,10240,16384,32768\"", file=f)
        print(file=f)
        print("rm -rf \"$SCRATCHDIR\"", file=f)
        print("echo \"DONE\"", file=f)
