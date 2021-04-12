import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np
import wandb
import sys
import itertools
from progressbar import progressbar
import os
import hashlib
import copy
import pickle
import gzip
import traceback

CACHE_DIR = 'D:\\runscache'
api = wandb.Api()
FIGSIZE=(6, 4)
POP_RENDER = [32, 128, 512, 2048, 10240, 32768]
pop_list = [32, 128, 200, 512, 1024, 2048, 5000, 10240, 16384, 32768]
linestyles = [':','-.','--','-']

class MyRun:
    def __init__(self, wandbrun, keep_history):
        self.name = wandbrun.name
        self.id = wandbrun.id
        self.sweep = None if wandbrun.sweep is None else {'id': wandbrun.sweep.id, 'name': wandbrun.sweep.config['name']}
        self.config = copy.deepcopy(dict(wandbrun.config))
        self.summary = copy.deepcopy(dict(wandbrun.summary))
        self._history = None if not keep_history else copy.deepcopy(wandbrun.history())

    def history(self):
        return self._history

    def __str__(self) -> str:
        return f"Run {self.name}({self.id})"

    @staticmethod
    def load_and_cache(filters, keep_history=False):
        h = hashlib.sha256(str(dict(**filters, **{'keep_history': keep_history})).encode('utf8'))
        if not os.path.exists(f'{CACHE_DIR}/{h.hexdigest()}.run'):
            runs = api.runs(f'kowalsky/thesis', filters=filters)
            runs = [MyRun(run, keep_history) for run in runs]
            with gzip.open(f'{CACHE_DIR}/{h.hexdigest()}.run', 'wb') as f:
                pickle.dump(runs, f)
        else:
            with gzip.open(f'{CACHE_DIR}/{h.hexdigest()}.run', 'rb') as f:
                runs = pickle.load(f)
        return runs


# render PSO2006 run times
for fn, in progressbar(list(itertools.product(
        [1, 7, 15, 19, 22, 24],
))):
    plt.figure(figsize=FIGSIZE)
    for idim, dim in enumerate([6,32,128,384]):
        runs = MyRun.load_and_cache({
            "$and": [
                {'config.run_type.value': 'time'},
                {'config.device.value': 'cpu'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2006'},
                {'config.pso.neigh.value': 'Random'},
                {'config.pop_size.value': {'$ne': None}},
                {'createdAt': {'$gte': '2021-04-09T22:04:00'}}
            ]
        })
        measure = np.zeros(len(pop_list))
        run_count = np.zeros(len(pop_list))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                i = pop_list.index(c['pop_size'])
                measure[i] += s['total_perf_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        plt.plot(pop_list, measure / run_count, label=f'CPU {dim}', linestyle=linestyles[idim], c='tab:blue')
        runs = MyRun.load_and_cache({
            "$and": [
                {'config.run_type.value': 'time'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2006'},
                {'config.pso.neigh.value': 'Random'},
                {'config.pop_size.value': {'$ne': None}},
            ]
        })
        measure = np.zeros(len(pop_list))
        run_count = np.zeros(len(pop_list))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                psize = c['pop_size']
                i = pop_list.index(c['pop_size'])
                measure[i] += s['total_proc_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        plt.plot(pop_list, measure / run_count, label=f'GPU {dim}', linestyle=linestyles[idim], c='tab:orange')
    plt.title(f"BBOB function $f_{{{fn}}}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(POP_RENDER)
    plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
    plt.xlim(32, 32768)
    plt.xlabel('Population size')
    plt.yticks([0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    plt.ylim(0.3, 10000.0)
    #plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.gca().get_yaxis().set_tick_params(which='minor', width=0)
    plt.ylabel('Running time [s]')
    plt.savefig(f"runs/time_pso2006_fn{fn}_alldim.pdf")
    plt.close()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle=':', c='black'),
    mlines.Line2D([0], [0], linestyle='-.', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], c='tab:blue'),
    mlines.Line2D([0], [0], c='tab:orange'),
], [
    'Problem dimension 6', 'Problem dimension 32', 'Problem dimension 128', 'Problem dimension 384',
    'CPU implementation', 'CUDA implementation',
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_pso2006_alldim_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)



# render PSO2011 run times
for fn, in progressbar(list(itertools.product(
        [1, 7, 15, 19, 22, 24],
))):
    plt.figure(figsize=FIGSIZE)
    for idim, dim in enumerate([6,32,128,384]):
        runs = MyRun.load_and_cache({
            "$and": [
                {'config.run_type.value': 'time'},
                {'config.device.value': 'cpu'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2011'},
                {'config.pso.neigh.value': 'Random'},
                {'config.pop_size': {'$neq': None}},
                {'config.pop_size.value': {'$ne': None}},
            ]
        })
        measure = np.zeros(len(pop_list))
        run_count = np.zeros(len(pop_list))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                i = pop_list.index(c['pop_size'])
                measure[i] += s['total_perf_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        plt.plot(pop_list, measure / run_count, label=f'CPU {dim}', linestyle=linestyles[idim], c='tab:blue')
        runs = MyRun.load_and_cache({
            "$and": [
                {'config.run_type.value': 'time,fitness'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2011'},
                {'config.pso.neigh.value': 'Random'},
                {'config.pop_size.value': {'$ne': None}},
            ]
        })
        measure = np.zeros(len(pop_list))
        run_count = np.zeros(len(pop_list))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                psize = c['pop_size']
                i = pop_list.index(c['pop_size'])
                measure[i] += s['total_proc_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        plt.plot(pop_list, measure / run_count, label=f'GPU {dim}', linestyle=linestyles[idim], c='tab:orange')
    plt.title(f"BBOB function $f_{{{fn}}}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(POP_RENDER)
    plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
    plt.xlim(32, 32768)
    plt.xlabel('Population size')
    plt.yticks([0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    plt.ylim(0.3, 10000.0)
    #plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.gca().get_yaxis().set_tick_params(which='minor', width=0)
    plt.ylabel('Running time [s]')
    plt.savefig(f"runs/time_pso2011_fn{fn}_alldim.pdf")
    plt.close()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle=':', c='black'),
    mlines.Line2D([0], [0], linestyle='-.', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], c='tab:blue'),
    mlines.Line2D([0], [0], c='tab:orange'),
], [
    'Problem dimension 6', 'Problem dimension 32', 'Problem dimension 128', 'Problem dimension 384',
    'CPU implementation', 'CUDA implementation',
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_pso2011_alldim_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
