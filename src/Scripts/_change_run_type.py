###############################
#
# Created by Patrik Valkovic
# 4/5/2021
#
###############################
import wandb
from progressbar import progressbar

api = wandb.Api()
#runs = api.runs(f'kowalsky/thesis', filters={
#    "$and": [
#        {'config.run_type.value': 'test'},
#        {'config.device.value': 'cpu'},
#        {'config.alg_group.value': 'es_mutation'},
#        {'createdAt': {'$gt': '2021-04-19T18:00:00Z'}}
#    ]
#})
#runs = list(runs)
#print(f"Going to change {len(runs)} runs")
#
#for run in progressbar(runs):
#    run.config['run_type'] = 'time'
#    run.update()


runs = api.runs(f'kowalsky/thesis', filters={
    "$and": [
        {'config.alg_group.value': 'es_schema'},
    ]
})
runs = list(runs)
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_failed'] = True
    run.update()
