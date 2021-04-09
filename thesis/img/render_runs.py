import matplotlib.pyplot as plt
import numpy as np
import wandb
import sys
api = wandb.Api()


# render PSO2006 run times
MEASURE='total_perf_time'
for fn in [1, 7, 15, 19, 22, 24]:
    for dim in [6,32,128,384]:
        runs = api.runs(f'kowalsky/thesis', filters={
            "$and": [
                {"config.run_type": 'time'},
                {'config.bbob_fn': fn},
                {'config.bbob_dim': dim},
                {'config.run_failed': False},
                {'config.alg_group': 'pso'},
                {'config.pso': {
                    'update': 'PSO2006',
                    'neigh': 'Random',
                }},
                {'State': {'$not': 'crashed'}}
            ]
        })
        cpujobs = list(filter(lambda r: r.config['device'] == 'cpu',runs))
        gpujobs = list(filter(lambda r: r.config['device'] == 'cuda',runs))
        pop_map = [32, 128, 200, 512, 1024, 2048, 5000, 10240, 16384, 32768]

        cpu_measures = np.zeros(len(pop_map))
        cpu_count = np.zeros(len(pop_map))
        for ji, j in enumerate(cpujobs):
            try:
                s, c = j.summary, j.config
                progress = s['iteration'] / s['max_iteration']
                proc_time = s[MEASURE] / progress
                i = pop_map.index(c['pop_size'])
                cpu_count[i] += 1
                cpu_measures[i] += proc_time
            except:
                print(sys.exc_info())
                print(j)
        cpu_mean = cpu_measures / cpu_count
        gpu_measures = np.zeros(len(pop_map))
        gpu_count = np.zeros(len(pop_map))
        for ji, j in enumerate(gpujobs):
            try:
                s, c = j.summary, j.config
                progress = s['iteration'] / s['max_iteration']
                proc_time = s[MEASURE] / progress
                i = pop_map.index(c['pop_size'])
                gpu_count[i] += 1
                gpu_measures[i] += proc_time
            except:
                print(sys.exc_info())
                print(j)
        gpu_mean = gpu_measures / gpu_count

        plt.plot(pop_map, cpu_mean, c='tab:blue')
        plt.plot(pop_map, gpu_mean, c='tab:orange')
        plt.title(f'Running time $f_{{{fn}}}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'_time_pso2006_fn{fn:02d}_d{dim}_perf.pdf')
        plt.close()
