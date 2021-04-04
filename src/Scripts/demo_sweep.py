###############################
#
# Created by Patrik Valkovic
# 4/4/2021
#
###############################
import os
import argparse
import wandb
from progressbar import progressbar

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = argparse.ArgumentParser()
p.add_argument("--first", type=int, help="First argument")
p.add_argument("--second_arg", type=float, help="Second argument")
p.add_argument('--repeat', type=int, help="Repeat index")
p.add_argument('--iterations', type=int, help="number of iterations")
args, _ = p.parse_known_args()


with wandb.init(config=args) as r:
    for i in range(args.iterations):
        r.log({
            'iteration': i,
            'repeat': args.repeat,
            'first': i + args.first,
            'second': i / args.second_arg
        }, step=i)
