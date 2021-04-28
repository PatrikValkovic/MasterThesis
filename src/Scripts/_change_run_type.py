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
                {'config.alg_group.value': 'ga_1'},
                {'config.run_type.value': 'time,fitness'},
                {'config.device.value': 'cuda'},
                {'config.pop_size.value': 32768},
                {'config.ga.elitism.value': False},
                {'config.sat.literals.value': 100},
            ]
        })
runs = list(runs)
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_type'] = 'time'
    run.update()
