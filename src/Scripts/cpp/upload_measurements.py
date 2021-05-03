###############################
#
# Created by Patrik Valkovic
# 5/3/2021
#
###############################
import wandb
import pandas as pd
from progressbar import progressbar

MEASUREMENT_FILE = 'D:\\cpp\\agg.out'
HEADER = ['popsize','literals','clauses','max_iteration','threads','iterations','time']

df = pd.read_csv(MEASUREMENT_FILE, sep=';', header=None)
df.columns = HEADER

for index, row in progressbar(list(df.iterrows())):
    with wandb.init(project='thesis', allow_val_change=True, config={
        'run_type': 'time',
        'run_failed': False,
        'problem_group': 'sat',
        'sat.literals': int(row['literals']),
        'sat.clauses': int(row['clauses']),
        'sat.mliter': 3,
        'sat.sliter': 0,
        'alg_group': 'ga_c',
        'pop_size': int(row['popsize']),
        'threads': int(row['threads']),
    }) as run:
        run.log({
            'max_iteration': int(row['max_iteration']),
            'iteration': int(row['iterations']),
            'total_real_time': row['time'],
        })
