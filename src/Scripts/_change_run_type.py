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
                {'config.run_failed.value': True},
                {'config.pop_size.value': 512},
                {'config.alg_group.value': 'es_crossover'},
                {'config.device.value': 'cpu'},
                {'config.run_type.value': 'time'},
                {'config.bbob_fn.value': 19},
                {'config.bbob_dim.value': 128},
                {'config.es.crossover.value': 'Arithmetic'},
            ]
        })
runs = list(runs)
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_failed'] = False
    run.update()
