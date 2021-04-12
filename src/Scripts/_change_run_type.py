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
        {'config.run_type.value': 'time'},
        {'config.device.value': 'cpu'},
        {'config.alg_group.value': 'pso'},
        {'config.pso.update.value': 'PSO2006'},
        {'config.pso.neigh.value': 'Random'},
        {'config.pop_size.value': {'$ne': None}},
        {'createdAt': {'$lt': '2021-04-09T22:04:00'}}
    ]
})
runs = list(runs)
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_type'] = 'test'
    run.update()
