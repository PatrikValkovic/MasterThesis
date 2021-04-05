###############################
#
# Created by Patrik Valkovic
# 4/5/2021
#
###############################
import wandb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

SWEEP_ID = "i7d0d5pr"
SPLIT_BY = "neigh_size"

api = wandb.Api()
sweep = api.sweep(f'kowalsky/thesis/{SWEEP_ID}')
runs = list(filter(lambda r: r.state == 'finished', sweep.runs))

def split_by(value):
    def _fn(runs):
        distinct = set(map(lambda r: r.config[value], runs))
        d = {str(k): [] for k in distinct}
        for r in runs:
            d[str(r.config[value])].append(r)
        return d
    return _fn

split = split_by(SPLIT_BY)(runs)

plt.figure(figsize=(12,8))
plt.title(f'Hyperparameters {SPLIT_BY}')
plt.yscale('log')
for ki, k in enumerate(split.keys()):
    max_iters = max(map(lambda x: x.summary['iteration'], split[k]))
    mean = np.zeros((len(split[k]), max_iters), dtype=float)
    q05 = np.zeros((len(split[k]), max_iters), dtype=float)
    q00 = np.zeros((len(split[k]), max_iters), dtype=float)
    for runi, run in enumerate(split[k]):
        hist = run.history(pandas=False)
        for step in range(max_iters):
            mean[runi, step] = hist[min(step, len(hist)-1)]['fitness_mean']
            q05[runi, step]  = hist[min(step, len(hist)-1)]['fitness_q05']
            q00[runi, step]  = hist[min(step, len(hist)-1)]['fitness_q00']
    mean = np.median(mean, axis=0)
    q05 = np.median(q05, axis=0)
    q00 = np.median(q00, axis=0)
    c = list(mcolors.TABLEAU_COLORS.values())[ki]
    plt.plot(range(max_iters), mean, c=c, linestyle='-', label=k)
    plt.plot(range(max_iters), q05, c=c, linestyle='--')
    plt.plot(range(max_iters), q00, c=c, linestyle=':')
plt.legend()
plt.show()
