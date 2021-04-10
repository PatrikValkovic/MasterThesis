import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import wandb
import sys
import itertools
from progressbar import progressbar
api = wandb.Api()


# render PSO2006 run times
pop_list = [32, 128, 200, 512, 1024, 2048, 5000, 10240, 16384, 32768]
for fn, dim in progressbar(list(itertools.product(
        [1, 7, 15, 19, 22, 24],
        [6,32,128,384]
))):
    plt.figure()
    runs = api.runs(f'kowalsky/thesis', filters={
        "$and": [
            {'config.run_type.value': 'time'},
            {'config.device.value': 'cpu'},
            {'config.bbob_fn.value': fn},
            {'config.bbob_dim.value': dim},
            {'config.alg_group.value': 'pso'},
            {'config.pso.update.value': 'PSO2011'},
        ]
    })
    perf_measure = np.zeros(len(pop_list))
    proc_measure = np.zeros(len(pop_list))
    run_count = np.zeros(len(pop_list))
    for run in runs:
        try:
            s, c = run.summary, run.config
            progress = s['iteration'] / s['max_iteration']
            psize = c['pop_size']
            i = pop_list.index(psize)
            perf, proc = s['total_perf_time'], s['total_proc_time']
            perf_measure[i] += perf
            proc_measure[i] += proc
            run_count[i] += 1
        except:
            print(sys.exc_info())
            print(run)
    plt.plot(pop_list, perf_measure / run_count, label='CPU perf', c='tab:blue', linestyle='--')
    plt.plot(pop_list, proc_measure / run_count, label='CPU proc', c='tab:blue')
    runs = api.runs(f'kowalsky/thesis', filters={
        "$and": [
            {'config.run_type.value': 'time,fitness'},
            {'config.device.value': 'cuda'},
            {'config.bbob_fn.value': fn},
            {'config.bbob_dim.value': dim},
            {'config.alg_group.value': 'pso'},
            {'config.pso.update.value': 'PSO2011'},
        ]
    })
    perf_measure = np.zeros(len(pop_list))
    proc_measure = np.zeros(len(pop_list))
    run_count = np.zeros(len(pop_list))
    for run in runs:
        try:
            s, c = run.summary, run.config
            progress = s['iteration'] / s['max_iteration']
            psize = c['pop_size']
            i = pop_list.index(psize)
            perf, proc = s['total_perf_time'], s['total_proc_time']
            perf_measure[i] += perf
            proc_measure[i] += proc
            run_count[i] += 1
        except Exception as e:
            print(e)
            print(run)
    plt.plot(pop_list, perf_measure / run_count, label='GPU perf', c='tab:orange', linestyle='--')
    plt.plot(pop_list, proc_measure / run_count, label='GPU proc', c='tab:orange')
    plt.legend()
    plt.xscale('log')
    plt.xticks([32, 128, 512, 2048, 10240, 32768])
    plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.yscale('log')
    plt.yticks([1, 3, 10, 30, 100, 300])
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.xlim(32, 32768)
    plt.savefig(f"_time_pso2011_fn{fn}_d{dim}.pdf")
