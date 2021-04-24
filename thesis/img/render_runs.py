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
import math

CACHE_DIR = '/tmp'
api = wandb.Api()
FIGSIZE=(6, 4)

#region support
class MyRun:
    def __init__(self, wandbrun, keep_history):
        self.name = wandbrun.name
        self.id = wandbrun.id
        self.sweep = None if wandbrun.sweep is None else {'id': wandbrun.sweep.id, 'name': wandbrun.sweep.config['name']}
        self.config = copy.deepcopy(dict(wandbrun.config))
        self.summary = copy.deepcopy(dict(wandbrun.summary))
        self._history = None if not keep_history else copy.deepcopy(list(wandbrun.scan_history()))

    def scan_history(self):
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
def new_group(name):
    exit()
    print(f"RUNNING {name}")
def round_plotup(val):
    base = math.floor(math.log10(val))
    round_to = 10 ** base
    return (int(val / round_to)+1) * round_to
def round_plotdown(val):
    base = math.floor(math.log10(val))
    round_to = 10 ** base
    return int(val / round_to) * round_to
def plot_generatelogticks(minval, maxval, ticks):
    diff = math.log(maxval) - math.log(minval)
    points = [math.exp(math.log(minval) + diff * s / ticks) for s in range(ticks+1)]
    points[0] = minval
    points[ticks] = maxval
    for i in range(1, ticks):
        base = math.floor(math.log10(points[i])) - 1
        round_to = 10 ** base
        points[i] = (int(points[i] / round_to)+1) * round_to
    return points
#endregion


#region ES mutation fitness
#new_group('ES mutation fitness')
NUM_Y_TICKS = 7
POP_SIZES=[32, 512, 10240, 32768]
MUTATIONS = ['AddFromNormal', 'AddFromCauchy', 'ReplaceUniform', 'AdaptiveStep']
PSIZE_C = {
    '32': 'tab:blue',
    '512': 'tab:orange',
    '10240': 'tab:green',
    '32768': 'tab:red',
}
STYLES=['-','--',':']
for fn, dim, mut in progressbar(list(itertools.product(
    [19, 22],
    [24, 128, 384],
    MUTATIONS,
))):
    plt.figure(figsize=FIGSIZE)
    maxval = -math.inf
    minval = math.inf
    for psize in POP_SIZES:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': psize},
                {'config.alg_group.value': 'es_mutation'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.mutation.value': mut}
            ]
        }, keep_history=True)
        runs = list(filter(lambda r: 'iteration' in r.summary, runs))
        print()
        print(f"fn {fn} with {dim} dim for {mut}, runs: {len(runs)} for pop size {psize}")
        print()
        max_iter = max(map(lambda r: r.summary['iteration'], runs))
        medians, q05, best = [], [], []
        for step in range(max_iter+1):
            cmedians, cq05, cbest = [], [], []
            for r in runs:
                h = r.scan_history()
                if len(h) <= step:
                    continue
                h = h[step]
                cmedians.append(h['fitness_median'])
                cq05.append(h['fitness_q05'])
                cbest.append(h['fitness_lowest'])
            if len(cmedians) <= 20:
                break
            medians.append(np.mean(cmedians))
            q05.append(np.mean(cq05))
            best.append(np.mean(cbest))
        minval = min(minval, np.min(best))
        maxval = max(maxval, np.max(medians))
        plt.plot(range(len(medians)), medians, c=PSIZE_C[str(psize)], linestyle=STYLES[0])
        plt.plot(range(len(q05)), q05,         c=PSIZE_C[str(psize)], linestyle=STYLES[1])
        plt.plot(range(len(best)), best,       c=PSIZE_C[str(psize)], linestyle=STYLES[2])
    plt.yscale('log')
    plt.gca().get_yaxis().clear()
    plt.xlim(0, 1000)
    plt.xlabel('Generation')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.minorticks_off()
    plt.ylabel('Objective function')
    plt.title(f"{mut} mutation of $f_{{{fn}}}$ with {dim} dimensions")
    plt.savefig(f'runs/fitness_es_mutation_f{fn}_dim{dim}_{mut}.pdf')
    plt.close()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    mlines.Line2D([0],[0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x), PSIZE_C.values())),
], [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"Population size {x}", PSIZE_C.keys()))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/fitness_es_mutation_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region ES schema fitness
new_group('ES schema fitness')
NUM_Y_TICKS = 7
POP_SIZES=[32, 512, 10240, 32768]
SCHEMAS = [
    {'name': 'ordinary', 'discard': False, 'replace': True},
    {'name': 'comma', 'discard': True, 'replace': True},
    {'name': 'plus', 'discard': False, 'replace': False},
]
PSIZE_C = {
    '32': 'tab:blue',
    '512': 'tab:orange',
    '10240': 'tab:green',
    '32768': 'tab:red',
}
STYLES=['-','--',':']
for fn, dim, schema in progressbar(list(itertools.product(
    [19, 22],
    [24, 128, 384],
    SCHEMAS,
))):
    plt.figure(figsize=FIGSIZE)
    maxval = -math.inf
    minval = math.inf
    for psize in POP_SIZES:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': psize},
                {'config.alg_group.value': 'es_schema'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.crossover.discard_parents.value': schema['discard']},
                {'config.es.crossover.replace_parents.value': schema['replace']},
            ]
        }, keep_history=True)
        runs = list(filter(lambda r: 'iteration' in r.summary, runs))
        print()
        print(f"fn {fn} with {dim} dim for {schema['name']}, runs: {len(runs)} for pop size {psize}")
        print()
        max_iter = max(map(lambda r: r.summary['iteration'], runs))
        medians, q05, best = [], [], []
        for step in range(max_iter+1):
            cmedians, cq05, cbest = [], [], []
            for r in runs:
                h = r.scan_history()
                if len(h) <= step:
                    continue
                h = h[step]
                cmedians.append(h['fitness_median'])
                cq05.append(h['fitness_q05'])
                cbest.append(h['fitness_lowest'])
            if len(cmedians) <= 20:
                break
            medians.append(np.mean(cmedians))
            q05.append(np.mean(cq05))
            best.append(np.mean(cbest))
        minval = min(minval, np.min(best))
        maxval = max(maxval, np.max(medians))
        plt.plot(range(len(medians)), medians, c=PSIZE_C[str(psize)], linestyle=STYLES[0])
        plt.plot(range(len(q05)), q05,         c=PSIZE_C[str(psize)], linestyle=STYLES[1])
        plt.plot(range(len(best)), best,       c=PSIZE_C[str(psize)], linestyle=STYLES[2])
    plt.yscale('log')
    plt.gca().get_yaxis().clear()
    plt.xlim(0, 1000)
    plt.xlabel('Generation')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.minorticks_off()
    plt.ylabel('Objective function')
    plt.title(f"{schema['name']} mutation of $f_{{{fn}}}$ with {dim} dimensions")
    plt.savefig(f'runs/fitness_es_schema_f{fn}_dim{dim}_{schema["name"]}.pdf')
    plt.close()
exit()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    mlines.Line2D([0],[0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x), PSIZE_C.values())),
], [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"Population size {x}", PSIZE_C.keys()))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/fitness_es_schema_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region ES schemas times
new_group('ES mutations schemas time')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
FNS= [19, 22]
DIMS= [24, 128, 384]
SCHEMAS = [
    {'name': 'ordinary', 'discard': False, 'replace': True, 'c': 'tab:blue'},
    {'name': 'comma', 'discard': True, 'replace': True, 'c': 'tab:orange'},
    {'name': 'plus', 'discard': False, 'replace': False, 'c': 'tab:green'},
]
TO_MEASURE = 'total_real_time'
for fn, dim in progressbar(list(itertools.product(
    FNS,
    DIMS,
))):
    plt.figure(figsize=FIGSIZE)
    maxval, minval = -math.inf, math.inf
    for schema in SCHEMAS:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': {'$in': POP_SIZES}},
                {'config.alg_group.value': 'es_schema'},
                {'config.device.value': 'cpu'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.crossover.discard_parents.value': schema['discard']},
                {'config.es.crossover.replace_parents.value': schema['replace']},
            ]
        })
        print()
        print(f"SCHEMA for {fn} fn with {dim} dim, schema {schema['name']} has {len(runs)} runs")
        measure = np.zeros(len(POP_SIZES))
        run_count = np.zeros(len(POP_SIZES))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                i = POP_SIZES.index(c['pop_size'])
                measure[i] += s[TO_MEASURE] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        measure = measure[run_count > 0] / run_count[run_count > 0]
        minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
        plt.plot(
            np.array(POP_SIZES)[run_count > 0],
            measure,
            c=schema['c'],
            linestyle='-',
        )

        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': {'$in': POP_SIZES}},
                {'config.alg_group.value': 'es_schema'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.crossover.discard_parents.value': schema['discard']},
                {'config.es.crossover.replace_parents.value': schema['replace']},
            ]
        })
        print(f"SCHEMA for {fn} fn with {dim} dim, schema {schema['name']} has {len(runs)} runs")
        measure = np.zeros(len(POP_SIZES))
        run_count = np.zeros(len(POP_SIZES))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                i = POP_SIZES.index(c['pop_size'])
                measure[i] += s[TO_MEASURE] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        measure = measure[run_count > 0] / run_count[run_count > 0]
        minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
        plt.plot(
            np.array(POP_SIZES)[run_count > 0],
            measure,
            c=schema['c'],
            linestyle='--',
        )
    plt.title(f"BBOB function $f_{{{fn}}}$ with ${dim}$ dimensions")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([32, 128, 512, 2048, 10240, 32768])
    plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
    plt.xlim(32, 32768)
    plt.xlabel('Population size')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.minorticks_off()
    plt.ylabel('Running time [s]')
    plt.savefig(f"runs/time_es_schema_fn{fn}_{dim}d.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x['c']), SCHEMAS)),
], [
    'CPU', 'CUDA',
    *list(map(lambda x: f'{x["name"]} schema', SCHEMAS))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_es_schema_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region ES mutation times
new_group('ES mutations running time')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
FNS= [19, 22]
DIMS= [24, 128, 384]
MUTATIONS = ['AddFromNormal', 'AddFromCauchy', 'ReplaceUniform', 'AdaptiveStep']
MUT_COLORS = {
    'AddFromNormal': 'tab:blue',
    'AddFromCauchy': 'tab:orange',
    'ReplaceUniform': 'tab:green',
    'AdaptiveStep': 'tab:red',
}
TO_MEASURE = 'total_real_time'
for fn, dim in progressbar(list(itertools.product(
    FNS,
    DIMS
))):
    plt.figure(figsize=FIGSIZE)
    maxval, minval = -math.inf, math.inf
    for mut in MUTATIONS:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': {'$in': POP_SIZES}},
                {'config.alg_group.value': 'es_mutation'},
                {'config.device.value': 'cpu'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.mutation.value': mut}
            ]
        })
        measure = np.zeros(len(POP_SIZES))
        run_count = np.zeros(len(POP_SIZES))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                i = POP_SIZES.index(c['pop_size'])
                measure[i] += s[TO_MEASURE] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        measure = measure[run_count > 0] / run_count[run_count > 0]
        minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
        plt.plot(
            np.array(POP_SIZES)[run_count > 0],
            measure,
            c=MUT_COLORS[mut],
            linestyle='-',
        )

        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': {'$in': POP_SIZES}},
                {'config.alg_group.value': 'es_mutation'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.mutation.value': mut}
            ]
        })
        measure = np.zeros(len(POP_SIZES))
        run_count = np.zeros(len(POP_SIZES))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                i = POP_SIZES.index(c['pop_size'])
                measure[i] += s[TO_MEASURE] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        measure = measure[run_count > 0] / run_count[run_count > 0]
        minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
        plt.plot(
            np.array(POP_SIZES)[run_count > 0],
            measure,
            c=MUT_COLORS[mut],
            linestyle='--',
        )
    plt.title(f"BBOB function $f_{{{fn}}}$ with ${dim}$ dimensions")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([32, 128, 512, 2048, 10240, 32768])
    plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
    plt.xlim(32, 32768)
    plt.xlabel('Population size')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.minorticks_off()
    plt.ylabel('Running time [s]')
    plt.savefig(f"runs/time_es_mutation_fn{fn}_{dim}d.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=MUT_COLORS[x]), MUTATIONS)),
], [
    'CPU', 'CUDA',
    *MUTATIONS
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_es_mutation_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region PSO neighborhood fitness
new_group('PSO neighborhood fitness')
NUM_Y_TICKS = 7
POP_SIZES=[121, 529, 4900, 22500]
PSIZE_C = {
    '121': 'tab:blue',
    '529': 'tab:orange',
    '4900': 'tab:green',
    '22500': 'tab:red',
}
NEIGHBORHOODS = [
    {'label': 'Random', 'selection': [
        {'config.pso.neigh.value': 'Random'},
    ], 'pop_sizes': POP_SIZES},
    {'label': 'Nearest', 'selection': [
        {'config.pso.neigh.value': 'Nearest'},
    ], 'pop_sizes': POP_SIZES[:3]},
    {'label': 'Circle', 'selection': [
        {'config.pso.neigh.value': 'Circle'},
    ], 'pop_sizes': POP_SIZES},
    {'label': 'Linear Grid', 'selection': [
        {'config.pso.neigh.value': 'Grid2D'},
        {'config.pso.neigh.subtype.value': 'linear'},
    ], 'pop_sizes': POP_SIZES[1:]},
    {'label': 'Compact Grid', 'selection': [
        {'config.pso.neigh.value': 'Grid2D'},
        {'config.pso.neigh.subtype.value': 'compact'},
    ], 'pop_sizes': POP_SIZES[1:]},
    {'label': 'Diamond Grid', 'selection': [
        {'config.pso.neigh.value': 'Grid2D'},
        {'config.pso.neigh.subtype.value': 'diamond'},
    ], 'pop_sizes': POP_SIZES[1:]},
]
STYLES=['-','--',':']
for fn, neig in progressbar(list(itertools.product(
    [1, 7, 15, 19, 22, 24],
    NEIGHBORHOODS
))):
    plt.figure(figsize=FIGSIZE)
    maxval = -math.inf
    minval = math.inf
    for psize in neig['pop_sizes']:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_type.value': 'time,fitness'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.pop_size.value': psize},
                {'config.bbob_dim.value': 128},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2006'},
                {'createdAt': {'$gte': '2021-04-09T22:04:00'}},
                {'config.run_failed.value': False},
                *neig['selection'],
            ]
        }, keep_history=True)
        runs = list(filter(lambda r: 'iteration' in r.summary, runs))
        print()
        print(neig)
        print(f"psize {psize}, {fn}fn, runs: {len(runs)}")
        print()
        max_iter = max(map(lambda r: r.summary['iteration'], runs))
        medians, q05, best = [], [], []
        for step in range(max_iter+1):
            cmedians, cq05, cbest = [], [], []
            for r in runs:
                h = r.scan_history()
                if len(h) <= step:
                    continue
                h = h[step]
                cmedians.append(h['fitness_median'])
                cq05.append(h['fitness_q05'])
                cbest.append(h['fitness_lowest'])
            if len(cmedians) <= 20:
                break
            medians.append(np.mean(cmedians))
            q05.append(np.mean(cq05))
            best.append(np.mean(cbest))
        minval = min(minval, np.min(best))
        maxval = max(maxval, np.max(medians))
        plt.plot(range(len(medians)), medians, c=PSIZE_C[str(psize)], linestyle=STYLES[0])
        plt.plot(range(len(q05)), q05,         c=PSIZE_C[str(psize)], linestyle=STYLES[1])
        plt.plot(range(len(best)), best,       c=PSIZE_C[str(psize)], linestyle=STYLES[2])
    plt.yscale('log')
    plt.gca().get_yaxis().clear()
    plt.xlim(0, 1000)
    plt.xlabel('Generation')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.minorticks_off()
    plt.ylabel('Objective function')
    plt.title(f"{neig['label']} neighborhood BBOB $f_{{{fn}}}$")
    plt.savefig(f'runs/fitness_pso_f{fn}_neigh{neig["label"].replace(" ","")}.pdf')
    plt.close()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    mlines.Line2D([0],[0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x), PSIZE_C.values())),
], [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"{x} particles", PSIZE_C.keys()))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/fitness_pso_neigh_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region PSO neighborhood times
new_group('PSO neighborhood times')
NUM_Y_TICKS = 7
POP_RENDER = [36,121,225,529,1225,2500,4900,10000,16900,22500]
NEIGHBORHOODS = [
    {'label': 'Random', 'selection': [
        {'config.pso.neigh.value': 'Random'},
    ], 'color': 'tab:blue'},
    {'label': 'Nearest', 'selection': [
        {'config.pso.neigh.value': 'Nearest'},
    ], 'color': 'tab:green'},
    {'label': 'Circle', 'selection': [
        {'config.pso.neigh.value': 'Circle'},
    ], 'color': 'tab:orange'},
    #{'label': 'Linear Grid', 'selection': [
    #    {'config.pso.neigh.value': 'Grid2D'},
    #    {'config.pso.neigh.subtype.value': 'linear'},
    #], 'color': 'tab:red'},
    #{'label': 'Compact Grid', 'selection': [
    #    {'config.pso.neigh.value': 'Grid2D'},
    #    {'config.pso.neigh.subtype.value': 'compact'},
    #], 'color': 'tab:purple'},
    {'label': 'Diamond Grid', 'selection': [
        {'config.pso.neigh.value': 'Grid2D'},
        {'config.pso.neigh.subtype.value': 'diamond'},
    ], 'color': 'tab:red'},
]
for fn, in progressbar(list(itertools.product(
    [1, 7, 15, 19, 22, 24],
))):
    plt.figure(figsize=FIGSIZE)
    maxval = -math.inf
    minval = math.inf
    for neig in NEIGHBORHOODS:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_type.value': 'time'},
                {'config.device.value': 'cpu'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': 128},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2006'},
                {'config.pop_size.value': {'$in': POP_RENDER}},
                {'createdAt': {'$gte': '2021-04-09T22:04:00'}},
                {'config.run_failed.value': False},
                *neig['selection'],
            ]
        })
        measure = np.zeros(len(POP_RENDER))
        run_count = np.zeros(len(POP_RENDER))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                i = POP_RENDER.index(c['pop_size'])
                measure[i] += s['total_perf_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        measure = measure[run_count > 0] / run_count[run_count > 0]
        minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
        plt.plot(
            np.array(POP_RENDER)[run_count > 0],
            measure,
            c=neig['color'],
            linestyle='-',
        )
        runs = MyRun.load_and_cache({
            "$and": [
                {'config.run_type.value': 'time,fitness'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': 128},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2006'},
                {'config.pop_size.value': {'$in': POP_RENDER}},
                {'createdAt': {'$gte': '2021-04-09T22:04:00'}},
                {'config.run_failed.value': False},
                *neig['selection'],
            ]
        })
        measure = np.zeros(len(POP_RENDER))
        run_count = np.zeros(len(POP_RENDER))
        for run in runs:
            try:
                s, c = run.summary, run.config
                if 'iteration' not in s:
                    continue
                progress = s['iteration'] / s['max_iteration']
                psize = c['pop_size']
                i = POP_RENDER.index(c['pop_size'])
                measure[i] += s['total_perf_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        measure = measure[run_count > 0] / run_count[run_count > 0]
        minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
        plt.plot(
            np.array(POP_RENDER)[run_count > 0],
            measure,
            c=neig['color'],
            linestyle='--',
        )
    plt.title(f"BBOB function $f_{{{fn}}}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([36, 121, 529, 2500, 10000, 22500])
    plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
    plt.xlim(36, 22500)
    plt.xlabel('Population size')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.minorticks_off()
    plt.ylabel('Running time [s]')
    plt.savefig(f"runs/time_pso2006_fn{fn}_neigh.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x['color']), NEIGHBORHOODS)),
], [
    'CPU', 'CUDA',
    *list(map(lambda x: x['label'], NEIGHBORHOODS))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_pso_neigh_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region PSO2011 performance
new_group('PSO2011 performance')
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
STYLES = ['-','--',':']
NUM_Y_TICKS = 7
for fn, in progressbar(list(itertools.product(
        [1, 7, 15, 19, 22, 24],
))):
    plt.figure(figsize=FIGSIZE)
    minval = math.inf
    maxval = -math.inf
    for ips, popsize in enumerate([32, 512, 10240, 32768]):
        runs = MyRun.load_and_cache({
            "$and": [
                {'config.run_type.value': 'time,fitness'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': 128},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2011'},
                {'config.pso.neigh.value': 'Random'},
                {'config.pop_size.value': popsize},
                {'config.run_failed.value': False},
            ]
        }, keep_history=True)
        medians = np.array(list(map(lambda r: list(map(lambda h: h['fitness_median'], r.scan_history())), runs)))
        medians = np.mean(medians, axis=0)
        q05 = np.array(list(map(lambda r: list(map(lambda h: h['fitness_q05'], r.scan_history())), runs)))
        q05 = np.mean(q05, axis=0)
        best = np.array(list(map(lambda r: list(map(lambda h: h['fitness_lowest'], r.scan_history())), runs)))
        best = np.mean(best, axis=0)
        minval = min(minval, best.min())
        maxval = max(maxval, medians.max())
        plt.plot(range(len(medians)), medians, c=COLORS[ips], linestyle=STYLES[0])
        plt.plot(range(len(q05)), q05, c=COLORS[ips], linestyle=STYLES[1])
        plt.plot(range(len(best)), best, c=COLORS[ips], linestyle=STYLES[2])
    plt.yscale('log')
    plt.gca().get_yaxis().clear()
    plt.xlim(0, 1000)
    plt.xlabel('Generation')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.minorticks_off()
    plt.ylabel('Objective function')
    plt.title(f"BBOB function $f_{{{fn}}}$")
    plt.savefig(f'runs/fitness_pso2011_f{fn}.pdf')
    plt.close()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle=':', c='black'),
    mlines.Line2D([0], [0], c='tab:blue'),
    mlines.Line2D([0], [0], c='tab:orange'),
    mlines.Line2D([0], [0], c='tab:green'),
    mlines.Line2D([0], [0], c='tab:red'),
], [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    '32 particles', '512 particles',
    '10240 particles', '32768 particles',
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/fitness_pso2011_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region PSO2006 performance
new_group('PSO2006 performance')
for fn, in progressbar(list(itertools.product(
        [1, 7, 15, 19, 22, 24],
))):
    plt.figure(figsize=FIGSIZE)
    minval = math.inf
    maxval = -math.inf
    for ips, popsize in enumerate([32, 512, 10240, 32768]):
        runs = MyRun.load_and_cache({
            "$and": [
                {'config.run_type.value': 'fitness'},
                {'config.device.value': 'cuda'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': 128},
                {'config.alg_group.value': 'pso'},
                {'config.pso.update.value': 'PSO2006'},
                {'config.pso.neigh.value': 'Random'},
                {'config.pop_size.value': popsize},
                {'config.run_failed.value': False},
            ]
        }, keep_history=True)
        medians = np.array(list(map(lambda r: list(map(lambda h: h['fitness_median'], r.scan_history())), runs)))
        medians = np.mean(medians, axis=0)
        q05 = np.array(list(map(lambda r: list(map(lambda h: h['fitness_q05'], r.scan_history())), runs)))
        q05 = np.mean(q05, axis=0)
        best = np.array(list(map(lambda r: list(map(lambda h: h['fitness_lowest'], r.scan_history())), runs)))
        best = np.mean(best, axis=0)
        minval = min(minval, best.min())
        maxval = max(maxval, medians.max())
        plt.plot(range(len(medians)), medians, c=COLORS[ips], linestyle=STYLES[0])
        plt.plot(range(len(q05)), q05, c=COLORS[ips], linestyle=STYLES[1])
        plt.plot(range(len(best)), best, c=COLORS[ips], linestyle=STYLES[2])
    plt.yscale('log')
    plt.gca().get_yaxis().clear()
    plt.xlim(0, 1000)
    plt.xlabel('Generation')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
    plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    plt.minorticks_off()
    plt.ylabel('Objective function')
    plt.title(f"BBOB function $f_{{{fn}}}$")
    plt.savefig(f'runs/fitness_pso2006_f{fn}.pdf')
    plt.close()
#endregion

#region PSO2006 run times
new_group('PSO2006 run times')
POP_RENDER = [32, 128, 512, 2048, 10240, 32768]
pop_list = [32, 128, 200, 512, 1024, 2048, 5000, 10240, 16384, 32768]
linestyles = [':','-.','--','-']
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
                {'config.pop_size.value': {'$in': pop_list}},
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
                {'config.pop_size.value': {'$in': pop_list}},
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
                measure[i] += s['total_perf_time'] / progress
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
#endregion

# region PSO2011 run times
new_group('PSO2011 run times')
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
                {'config.pop_size.value': {'$in': pop_list}},
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
                {'config.pop_size.value': {'$in': pop_list}},
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
                measure[i] += s['total_perf_time'] / progress
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
#endregion
