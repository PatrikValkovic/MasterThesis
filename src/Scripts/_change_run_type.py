###############################
#
# Created by Patrik Valkovic
# 4/5/2021
#
###############################
import wandb
from progressbar import progressbar

api = wandb.Api()
runs = api.runs(f'kowalsky/thesis', filters={
    "$and": [
        {'state': 'finished'},
        {'config.run_failed.value': False},
    ]
})
runs = list(filter(lambda x: 'iteration' not in x.summary, runs))
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_failed'] = True

