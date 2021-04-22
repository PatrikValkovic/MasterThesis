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
        {'config.run_failed.value': False},
        {'config.pop_size.value': {'$lte': 2048}},
        {'config.alg_group.value': 'es_mutation'},
        {'config.device.value': 'cuda'},
        {'config.bbob_fn.value': 19},
        {'config.bbob_dim.value': 384},
        {'config.es.mutation.value': {'$in': ['AddFromNormal', 'AddFromCauchy', 'ReplaceUniform']}},
    ]
})
runs = list(runs)
print(f"Going to change {len(runs)} runs")

for run in progressbar(runs):
    run.config['run_failed'] = True
    run.update()
