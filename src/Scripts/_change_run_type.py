###############################
#
# Created by Patrik Valkovic
# 4/5/2021
#
###############################
import wandb
import progressbar


api = wandb.Api()
runs = api.runs(f'kowalsky/thesis', filters={
    "$or": [
        {"config.run_type": 'testing'},
        {"config.run_type": None},
    ]
})
print(f"Going to change {len(runs)} runs")

for run in progressbar.progressbar(runs):
    run.config['run_type'] = 'test'
    run.update()

