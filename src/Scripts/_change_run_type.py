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
        {'config.alg_group.value': 'ga_selection'},
        {'config.run_type.value': 'time'},
        {'config.device.value': 'cpu'},
        {'config.ga.selection.value': 'StochasticUniversalSampling'},
    ]
})
print(f"Going to change {len(runs)} runs")
runs = list(runs)

for run in progressbar(runs):
    run.config['run_failed'] = True
    run.update()
