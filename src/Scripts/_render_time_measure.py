###############################
#
# Created by Patrik Valkovic
# 02.04.21
#
###############################
import wandb
import numpy as np
import matplotlib.pyplot as plt


api = wandb.Api()
runs = api.runs('thesis')
iteration_proc_time = []
iteration_perf_time = []

for run in runs:
    if run.state == "finished":
       iterproctime = run.history()['iteration_proc_time'].to_numpy()
       iteration_proc_time.append(iterproctime)
       iterperftime = run.history()['iteration_perf_time'].to_numpy()
       iteration_perf_time.append(iterperftime)

iteration_proc_time = np.concatenate(iteration_proc_time, axis=0)
iteration_perf_time = np.concatenate(iteration_perf_time, axis=0)


plt.hist(iteration_proc_time, bins=20)
plt.title('Proc time')
plt.show()
plt.hist(iteration_perf_time, bins=20)
plt.title('Perf time')
plt.show()
