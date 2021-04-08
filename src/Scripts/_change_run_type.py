###############################
#
# Created by Patrik Valkovic
# 4/5/2021
#
###############################
import wandb


api = wandb.Api()
runs = api.runs(f'kowalsky/thesis', filters={
    "$or": [
        {"config.run_type": 'testing'},
        {"config.run_type": None},
    ]
})

for run in runs:
    run.config['run_type'] = 'test'
    run.update()
