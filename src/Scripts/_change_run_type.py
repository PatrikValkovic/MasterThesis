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
        {'config.alg_group.value': 'es_crossover'},
        {'config.device.value': 'cpu'},
        {'config.run_type.value': 'test'},
        {'config.pop_size.value': {'$in': [32,128,200,512,1024,2048,5000,10240,16384,32768]}},
    ]
})
#runs = list(runs)
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_type'] = 'time'
    run.update()
