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
        {'createdAt': {'$gt': '2021-04-12T09:20:59Z'}}
    ]
})
runs = list(runs)
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_failed'] = True
    run.update()
