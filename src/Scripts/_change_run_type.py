###############################
#
# Created by Patrik Valkovic
# 4/5/2021
#
###############################
import wandb


api = wandb.Api()
runs = api.runs(f'kowalsky/thesis', filters={
    "$and": [
        {"config.run_type": {"$gt": "2021-04-07T11:00:00"}},
    ]
})

for run in runs:
    run.config['run_type'] = 'test'
    run.config['run_failed'] = True
    run.update()