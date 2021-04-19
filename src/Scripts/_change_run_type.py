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
        {'config.run_type.value': 'test'},
        {'config.device.value': 'cpu'},
        {'config.alg_group.value': 'es_mutation'},
    ]
})
runs = list(runs)
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_type'] = 'time'
    run.update()
