###############################
#
# Created by Patrik Valkovic
# 28/04/2021
#
###############################
import itertools
import wandb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

SWEEPS = ["raz9xa1k",'q7879ej1','p2fc792w']
POPSIZES = [32,128,200,512,1024,2048,5000,10240,16384]
SCALES = ['Pipe','MultiplicativeInverse','RankScale','ExponentialScale','LogScale','LinearScale']
DEVICES = ['cpu','cuda']
STYLES = {'cpu':'-','cuda':'--'}
COLORS = {
    'Pipe': 'tab:blue',
    'MultiplicativeInverse': 'tab:orange',
    'RankScale': 'tab:green',
    'ExponentialScale': 'tab:red',
    'LogScale': 'tab:purple',
    'LinearScale': 'tab:gray'
}

api = wandb.Api()
for sweepid in SWEEPS:
    plt.figure(figsize=(12,8))
    for scale, dev in itertools.product(SCALES, DEVICES):
        times = np.zeros(len(POPSIZES))
        counts = np.zeros(len(POPSIZES))
        for i in range(len(POPSIZES)):
            runs = api.runs(f'kowalsky/thesis', filters={
                '$and': [
                    {'sweep': sweepid},
                    {'config.pop_size.value': POPSIZES[i]},
                    {'config.scale.value': scale},
                    {'config.device.value': dev}
                ]
            })
            for run in runs:
                times[i] += run.summary['total_real_time'] / (run.summary['iteration'] / run.summary['max_iteration'])
                counts[i] += 1
        plt.plot(POPSIZES, times / counts, c=COLORS[scale], linestyle=STYLES[dev], label=f'{dev} {scale}')
    plt.yscale('log')
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.legend()
    plt.savefig(f'SCALE_{sweepid}.pdf')
    plt.close()
