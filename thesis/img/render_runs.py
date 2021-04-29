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

from tensorflow.lite.python import lite

CACHE_DIR = 'D:\\runscache'
api = wandb.Api()
FIGSIZE=(6, 4)
FIGISZE_BIG=(12,8)

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
    def load_and_cache(filters, keep_history=False, reporting_metric=False, restore_files=False):
        as_str = str(dict(**filters, **{'keep_history': keep_history}))
        h = hashlib.sha256(as_str.encode('utf8'))
        if not os.path.exists(f'{CACHE_DIR}/{h.hexdigest()}.run') or restore_files:
            runs = api.runs(f'kowalsky/thesis', filters=filters)
            runs = [MyRun(run, keep_history) for run in runs]
            with gzip.open(f'{CACHE_DIR}/{h.hexdigest()}.run', 'wb') as f:
                pickle.dump(runs, f)
        else:
            with gzip.open(f'{CACHE_DIR}/{h.hexdigest()}.run', 'rb') as f:
                runs = pickle.load(f)
        if reporting_metric:
            print(f"\n{h.hexdigest()}.run for\n{as_str}\nHas {len(runs)} runs")
        return runs
def new_group(name):
    exit()
    print(f"RUNNING {name}")
def round_plotup(val):
    base = math.floor(math.log10(val))
    round_to = 10 ** base
    return (int(val / round_to)+1) * round_to
def round_plotdown(val):
    base = math.floor(math.log10(val) if val > 0 else 0)
    round_to = 10 ** base
    return int(val / round_to) * round_to
def plot_generatelogticks(minval, maxval, ticks):
    lm = math.log(minval) if minval > 0 else 0
    diff = math.log(maxval) - lm
    points = [math.exp(lm + diff * s / ticks) for s in range(ticks+1)]
    points[0] = minval
    points[ticks] = maxval
    for i in range(1, ticks):
        base = math.floor(math.log10(points[i])) - 1
        round_to = 10 ** base
        points[i] = (int(points[i] / round_to)+1) * round_to
    return points
#endregion


#region GA normal and elitism fitness
#new_group('GA fitness')
REFETCH=True
NUM_Y_TICKS = 7
POP_SIZES=[32,512,5000,10240,32768]
VARIABLES=[100, 300, 800, 1000, 2000, 5000]
PSIZE_C = {
    '32': 'tab:blue',
    '512': 'tab:orange',
    '5000': 'tab:green',
    '10240': 'tab:red',
    '32768': 'tab:gray',
}
STYLES=['-','-',':']
ELITISM=[False,True]
for vars,elitism in progressbar(list(itertools.product(
    VARIABLES, ELITISM
))):
    plt.figure(figsize=FIGSIZE)
    maxval = -math.inf
    minval = math.inf
    iters_render = -1
    for psize in POP_SIZES:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state' : 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': psize},
                {'config.alg_group.value': 'ga_1'},
                {'config.run_type.value': 'fitness'},
                {'config.ga.elitism.value': elitism},
                {'config.sat.literals.value': vars}
            ]
        }, keep_history=True, restore_files=REFETCH, reporting_metric=True)
        runs = list(filter(lambda r: 'iteration' in r.summary, runs))
        if len(runs) < 10:
            print(f"\nWARNING: {len(runs)} runs for GA fitness with {vars} variables and {psize} individuals")
            continue
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
            medians.append(np.mean(cmedians))
            q05.append(np.mean(cq05))
            best.append(np.mean(cbest))
        minval = min(minval, np.min(best))
        maxval = max(maxval, np.max(medians))
        best = np.array(best)
        if np.any(best == 0):
            iters_render = max(iters_render, np.where(best == 0)[0].min())
        #plt.plot(range(len(medians)), medians, c=PSIZE_C[str(psize)], linestyle=STYLES[0])
        plt.plot(range(len(q05)), q05,         c=PSIZE_C[str(psize)], linestyle=STYLES[1])
        #plt.plot(range(len(best)), best,       c=PSIZE_C[str(psize)], linestyle=STYLES[2])
    if minval == math.inf:
        print(f"\nWARNING GA fitness with {vars} literals has no measurements")
        continue
    if iters_render == -1:
        iters_render = 1000
    plt.xlim(0, iters_render)
    plt.xlabel('Generation')
    minval = round_plotdown(minval)
    maxval = round_plotup(maxval)
    plt.ylim(minval, maxval)
    plt.ylabel('Objective function')
    plt.title(f"GA fitness of 3SAT{' with elitism' if elitism else ''} problem with {vars} literals")
    plt.savefig(f'runs/fitness_ga{"_elitism" if elitism else ""}_3SAT_d{vars}.pdf')
    plt.close()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    #mlines.Line2D([0],[0], linestyle='-', c='black'),
    #mlines.Line2D([0],[0], linestyle='--', c='black'),
    #mlines.Line2D([0],[0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x), PSIZE_C.values())),
], [
    #'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"Population size {x}", PSIZE_C.keys()))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/fitness_ga_3SAT_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion
exit()

#region GA selection times
#new_group('GA selection time')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
SELECTIONS = [
    {'label': 'Tournament selection', 'key': 'Tournament', 'c': 'tab:blue'},
    {'label': 'Roulette selection', 'key': 'Roulette', 'c': 'tab:orange'},
    {'label': 'Stochastic Universal Sampling', 'key': 'StochasticUniversalSampling', 'c': 'tab:green'},
]
TO_MEASURE = 'total_real_time'
plt.figure(figsize=FIGISZE_BIG)
maxval, minval = -math.inf, math.inf
for selection in progressbar(SELECTIONS):
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.alg_group.value': 'ga_selection'},
            {'config.device.value': 'cpu'},
            {'config.run_type.value': 'time'},
            {'config.pop_size.value': {'$in': POP_SIZES}},
            {'config.ga.selection.value': selection['key']},
        ]
    }, reporting_metric=True)
    measure = np.zeros(len(POP_SIZES))
    run_count = np.zeros(len(POP_SIZES))
    for run in runs:
        try:
            s, c = run.summary, run.config
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
        c=selection['c'],
        linestyle='-',
    )
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.alg_group.value': 'ga_selection'},
            {'config.device.value': 'cuda'},
            {'config.run_type.value': 'time'},
            {'config.pop_size.value': {'$in': POP_SIZES}},
            {'config.ga.selection.value': selection['key']},
        ]
    }, reporting_metric=True)
    measure = np.zeros(len(POP_SIZES))
    run_count = np.zeros(len(POP_SIZES))
    for run in runs:
        try:
            s, c = run.summary, run.config
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
        c=selection['c'],
        linestyle='--',
    )
plt.title(f"Running time of selection operators")
plt.xscale('log')
plt.yscale('log')
plt.xticks([32, 128, 512, 2048, 10240, 32768])
plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim(32, 32768)
plt.xlabel('Literals')
minval = round_plotdown(minval)
maxval = round_plotup(maxval)
plt.ylim(minval, maxval)
plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
plt.minorticks_off()
plt.ylabel('Running time [s]')
plt.savefig(f"runs/time_ga_selections.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='k'),
    mlines.Line2D([0],[0], linestyle='-', c='k'),
    *(list(map(lambda s: mlines.Line2D([0],[0], linestyle='-', c=s['c']), SELECTIONS)))
], [
    'CPU', 'CUDA',
    *(list(map(lambda s: s['label'], SELECTIONS)))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_ga_selections_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion


#region GA scale times
new_group('GA scale time')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
LITERALS = [100,1000]
SCALES = [
    {'label': 'Linear', 'key': 'LinearScale', 'c': 'tab:blue'},
    {'label': 'Logarithmic', 'key': 'LogScale', 'c': 'tab:orange'},
    {'label': 'Exponential', 'key': 'ExponentialScale', 'c': 'tab:green'},
    {'label': 'Rank', 'key': 'RankScale', 'c': 'tab:red'},
    {'label': 'Inverse', 'key': 'MultiplicativeInverse', 'c': 'tab:purple'},
    {'label': 'None', 'key': 'Pipe', 'c': 'tab:gray'},
]
TO_MEASURE = 'total_real_time'
for liters in LITERALS:
    plt.figure(figsize=FIGSIZE)
    maxval, minval = -math.inf, math.inf
    for scale, in progressbar(list(itertools.product(
        SCALES
    ))):
        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': {'$in': POP_SIZES}},
                {'config.alg_group.value': 'ga_scaling'},
                {'config.device.value': 'cpu'},
                {'config.run_type.value': 'time'},
                {'config.ga.elitism.value': False},
                {'config.scale.value': scale['key']},
                {'config.sat.literals.value': liters},
            ]
        })
        if len(runs) == 0:
            print(f"\nWARNING, 0 runs for CPU GA SCALING with {liters} literals and {scale['label']} scale")
            continue
        measure = np.zeros(len(POP_SIZES))
        run_count = np.zeros(len(POP_SIZES))
        for run in runs:
            try:
                s, c = run.summary, run.config
                progress = s['iteration'] / s['max_iteration']
                i = POP_SIZES.index(c['pop_size'])
                measure[i] += s[TO_MEASURE] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        measure = measure[run_count > 0] / run_count[run_count > 0]
        if np.any(run_count == 0):
            print(f"\nWARNING, missing measurements for CPU GA SCALING with {liters} literals and {scale['label']} for pop sizes {list(map(lambda x: POP_SIZES[x], np.where(run_count == 0)[0].tolist()))}")
        minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
        plt.plot(
            np.array(POP_SIZES)[run_count > 0],
            measure,
            c=scale['c'],
            linestyle='-',
        )

        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': {'$in': POP_SIZES}},
                {'config.alg_group.value': 'ga_scaling'},
                {'config.device.value': 'cuda'},
                {'config.run_type.value': 'time'},
                {'config.ga.elitism.value': False},
                {'config.scale.value': scale['key']},
                {'config.sat.literals.value': liters},
            ]
        })
        if len(runs) == 0:
            print(f"\nWARNING, 0 runs for GPU GA SCALING with {liters} literals and {scale['label']} scale")
            continue
        measure = np.zeros(len(POP_SIZES))
        run_count = np.zeros(len(POP_SIZES))
        for run in runs:
            try:
                s, c = run.summary, run.config
                progress = s['iteration'] / s['max_iteration']
                i = POP_SIZES.index(c['pop_size'])
                measure[i] += s[TO_MEASURE] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        measure = measure[run_count > 0] / run_count[run_count > 0]
        minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
        if np.any(run_count == 0):
            print(f"\nWARNING, missing measurements for GPU GA SCALING with {liters} literals and {scale['label']} for pop sizes {list(map(lambda x: POP_SIZES[x], np.where(run_count == 0)[0].tolist()))}")
        plt.plot(
            np.array(POP_SIZES)[run_count > 0],
            measure,
            c=scale['c'],
            linestyle='--',
        )
    plt.title(f"Scale functions running times for {liters} literals")
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
    plt.savefig(f"runs/time_ga_scale_{liters}l.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x['c']), SCALES)),
], [
    'CPU', 'CUDA',
    *list(map(lambda x: f'{x["label"]} scale', SCALES))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_ga_scale_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region GA elitism and normal times
new_group('GA elitism time')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
VARIABLES = [100, 300, 800, 2000]
VAR_C = {
    '100': 'tab:blue',
    '300': 'tab:orange',
    '800': 'tab:green',
    '2000': 'tab:red',
}
TO_MEASURE = 'total_real_time'
plt.figure(figsize=FIGSIZE)
maxval, minval = 2000.0, 0.6
for vars, in progressbar(list(itertools.product(
    VARIABLES
))):
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.pop_size.value': {'$in': POP_SIZES}},
            {'config.alg_group.value': 'ga_1'},
            {'config.device.value': 'cpu'},
            {'config.run_type.value': 'time'},
            {'config.ga.elitism.value': True},
            {'config.sat.literals.value': vars}
        ]
    })
    measure = np.zeros(len(POP_SIZES))
    run_count = np.zeros(len(POP_SIZES))
    for run in runs:
        try:
            s, c = run.summary, run.config
            progress = s['iteration'] / s['max_iteration']
            i = POP_SIZES.index(c['pop_size'])
            measure[i] += s[TO_MEASURE] / progress
            run_count[i] += 1
        except:
            traceback.print_exc()
            print(run)
    measure = measure[run_count > 0] / run_count[run_count > 0]
    for pindex in np.where(run_count==0)[0]:
        print(f"\nWARNING GA ELITISM CPU fitness for {vars} literals with {POP_SIZES[pindex]} has no measurement")
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c=VAR_C[str(vars)],
        linestyle='-',
    )

    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.pop_size.value': {'$in': POP_SIZES}},
            {'config.alg_group.value': 'ga_1'},
            {'config.device.value': 'cuda'},
            {'$or': [
                {'config.run_type.value': 'time,fitness'},
                {'config.run_type.value': 'time'},
            ]},
            {'config.ga.elitism.value': True},
            {'config.sat.literals.value': vars}
        ]
    })
    measure = np.zeros(len(POP_SIZES))
    run_count = np.zeros(len(POP_SIZES))
    for run in runs:
        try:
            s, c = run.summary, run.config
            progress = s['iteration'] / s['max_iteration']
            i = POP_SIZES.index(c['pop_size'])
            measure[i] += s[TO_MEASURE] / progress
            run_count[i] += 1
        except:
            traceback.print_exc()
            print(run)
    measure = measure[run_count > 0] / run_count[run_count > 0]
    for pindex in np.where(run_count==0)[0]:
        print(f"\nWARNING GA ELITISM GPU fitness for {vars} literals with {POP_SIZES[pindex]} has no measurement")
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c=VAR_C[str(vars)],
        linestyle='--',
    )
plt.title(f"Genetic algorithm with elitism for 3SAT problem")
plt.xscale('log')
plt.yscale('log')
plt.xticks([32, 128, 512, 2048, 10240, 32768])
plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim(32, 32768)
plt.xlabel('Population size')
plt.ylim(minval, maxval)
plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter())
plt.minorticks_off()
plt.ylabel('Running time [s]')
plt.savefig(f"runs/time_ga_elitism.pdf")

new_group('GA time')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
VARIABLES = [100, 300, 800, 2000]
VAR_C = {
    '100': 'tab:blue',
    '300': 'tab:orange',
    '800': 'tab:green',
    '2000': 'tab:red',
}
TO_MEASURE = 'total_real_time'
plt.figure(figsize=FIGSIZE)
for vars, in progressbar(list(itertools.product(
    VARIABLES
))):
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.pop_size.value': {'$in': POP_SIZES}},
            {'config.alg_group.value': 'ga_1'},
            {'config.device.value': 'cpu'},
            {'config.run_type.value': 'time'},
            {'config.ga.elitism.value': False},
            {'config.sat.literals.value': vars}
        ]
    })
    measure = np.zeros(len(POP_SIZES))
    run_count = np.zeros(len(POP_SIZES))
    for run in runs:
        try:
            s, c = run.summary, run.config
            progress = s['iteration'] / s['max_iteration']
            i = POP_SIZES.index(c['pop_size'])
            measure[i] += s[TO_MEASURE] / progress
            run_count[i] += 1
        except:
            traceback.print_exc()
            print(run)
    measure = measure[run_count > 0] / run_count[run_count > 0]
    for pindex in np.where(run_count==0)[0]:
        print(f"\nWARNING GA CPU fitness for {vars} literals with {POP_SIZES[pindex]} has no measurement")
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c=VAR_C[str(vars)],
        linestyle='-',
    )

    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.pop_size.value': {'$in': POP_SIZES}},
            {'config.alg_group.value': 'ga_1'},
            {'config.device.value': 'cuda'},
            {'$or': [
                {'config.run_type.value': 'time,fitness'},
                {'config.run_type.value': 'time'},
            ]},
            {'config.ga.elitism.value': False},
            {'config.sat.literals.value': vars}
        ]
    })
    measure = np.zeros(len(POP_SIZES))
    run_count = np.zeros(len(POP_SIZES))
    for run in runs:
        try:
            s, c = run.summary, run.config
            progress = s['iteration'] / s['max_iteration']
            i = POP_SIZES.index(c['pop_size'])
            measure[i] += s[TO_MEASURE] / progress
            run_count[i] += 1
        except:
            traceback.print_exc()
            print(run)
    measure = measure[run_count > 0] / run_count[run_count > 0]
    for pindex in np.where(run_count==0)[0]:
        print(f"\nWARNING GA GPU fitness for {vars} literals with {POP_SIZES[pindex]} has no measurement")
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c=VAR_C[str(vars)],
        linestyle='--',
    )
plt.title(f"Genetic algorithm for 3SAT problem")
plt.xscale('log')
plt.yscale('log')
plt.xticks([32, 128, 512, 2048, 10240, 32768])
plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim(32, 32768)
plt.xlabel('Population size')
plt.ylim(minval, maxval)
plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter())
plt.minorticks_off()
plt.ylabel('Running time [s]')
plt.savefig(f"runs/time_ga.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x), VAR_C.values())),
], [
    'CPU', 'CUDA',
    *list(map(lambda x: f'{x} literals', VAR_C.keys()))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_ga_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region GA clausecount times
new_group('GA clausecount time')
print('Clausecount')
NUM_Y_TICKS = 7
CLAUSES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
TO_MEASURE = 'total_real_time'
plt.figure(figsize=FIGSIZE)
maxval, minval = -math.inf, math.inf
runs = MyRun.load_and_cache({
    "$and": [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.alg_group.value': 'ga_clausecount'},
        {'config.device.value': 'cpu'},
        {'config.run_type.value': 'time'},
    ]
})
measure = np.zeros(len(CLAUSES))
run_count = np.zeros(len(CLAUSES))
for run in runs:
    try:
        s, c = run.summary, run.config
        progress = s['iteration'] / s['max_iteration']
        i = CLAUSES.index(c['sat.clauses'])
        measure[i] += s[TO_MEASURE] / progress
        run_count[i] += 1
    except:
        traceback.print_exc()
        print(run)
measure = measure[run_count > 0] / run_count[run_count > 0]
minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
plt.plot(
    np.array(CLAUSES)[run_count > 0],
    measure,
    c='tab:blue',
    linestyle='-',
)
runs = MyRun.load_and_cache({
    "$and": [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.alg_group.value': 'ga_clausecount'},
        {'config.device.value': 'cuda'},
        {'config.run_type.value': 'time'},
    ]
})
measure = np.zeros(len(CLAUSES))
run_count = np.zeros(len(CLAUSES))
for run in runs:
    try:
        s, c = run.summary, run.config
        progress = s['iteration'] / s['max_iteration']
        i = CLAUSES.index(c['sat.clauses'])
        measure[i] += s[TO_MEASURE] / progress
        run_count[i] += 1
    except:
        traceback.print_exc()
        print(run)
measure = measure[run_count > 0] / run_count[run_count > 0]
minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
plt.plot(
    np.array(CLAUSES)[run_count > 0],
    measure,
    c='tab:orange',
    linestyle='-',
)
plt.title(f"Running time of 3SAT")
plt.xscale('log')
plt.yscale('log')
plt.xticks([32, 128, 512, 2048, 10240, 32768])
plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim(32, 32768)
plt.xlabel('Clauses')
minval = round_plotdown(minval)
maxval = round_plotup(maxval)
plt.ylim(minval, maxval)
plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
plt.minorticks_off()
plt.ylabel('Running time [s]')
plt.savefig(f"runs/time_ga_clausecount.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='tab:blue'),
    mlines.Line2D([0],[0], linestyle='-', c='tab:orange'),
], [
    'CPU', 'CUDA',
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_ga_clausecount_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region GA varcount times
new_group('GA varcount time')
print('Varcount')
NUM_Y_TICKS = 7
VARS = [32,128,200,512,1024,2048,5000,10240,16384,32768]
TO_MEASURE = 'total_real_time'
plt.figure(figsize=FIGSIZE)
maxval, minval = -math.inf, math.inf
runs = MyRun.load_and_cache({
    "$and": [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.alg_group.value': 'ga_varcount'},
        {'config.device.value': 'cpu'},
        {'config.run_type.value': 'time'},
    ]
})
measure = np.zeros(len(VARS))
run_count = np.zeros(len(VARS))
for run in runs:
    try:
        s, c = run.summary, run.config
        progress = s['iteration'] / s['max_iteration']
        i = VARS.index(c['sat.literals'])
        measure[i] += s[TO_MEASURE] / progress
        run_count[i] += 1
    except:
        traceback.print_exc()
        print(run)
measure = measure[run_count > 0] / run_count[run_count > 0]
minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
plt.plot(
    np.array(VARS)[run_count > 0],
    measure,
    c='tab:blue',
    linestyle='-',
)
runs = MyRun.load_and_cache({
    "$and": [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.alg_group.value': 'ga_varcount'},
        {'config.device.value': 'cuda'},
        {'config.run_type.value': 'time'},
    ]
})
measure = np.zeros(len(VARS))
run_count = np.zeros(len(VARS))
for run in runs:
    try:
        s, c = run.summary, run.config
        progress = s['iteration'] / s['max_iteration']
        i = VARS.index(c['sat.literals'])
        measure[i] += s[TO_MEASURE] / progress
        run_count[i] += 1
    except:
        traceback.print_exc()
        print(run)
measure = measure[run_count > 0] / run_count[run_count > 0]
minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
plt.plot(
    np.array(VARS)[run_count > 0],
    measure,
    c='tab:orange',
    linestyle='-',
)
plt.title(f"Running time of 3SAT")
plt.xscale('log')
plt.yscale('log')
plt.xticks([32, 128, 512, 2048, 10240, 32768])
plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim(32, 32768)
plt.xlabel('Literals')
minval = round_plotdown(minval)
maxval = round_plotup(maxval)
plt.ylim(minval, maxval)
plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
plt.minorticks_off()
plt.ylabel('Running time [s]')
plt.savefig(f"runs/time_ga_varcount.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='tab:blue'),
    mlines.Line2D([0],[0], linestyle='-', c='tab:orange'),
], [
    'CPU', 'CUDA',
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_ga_varcount_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region ES crossover fitness
new_group('ES crossover fitness')
NUM_Y_TICKS = 7
POP_SIZES=[32,512,10240,32768]
FNS= [19, 24]
DIMS= [24, 128, 384]
PSIZE_C = {
    '32': 'tab:blue',
    '512': 'tab:orange',
    '10240': 'tab:green',
    '32768': 'tab:red',
}
CROSSOVERS = [
    {'label': 'Arithmetic','crossover': 'Arithmetic'},
    {'label': 'Blend','crossover': 'Blend'},
    {'label': 'One-point','crossover': 'OnePoint1D'},
    {'label': 'Two-point','crossover': 'TwoPoint1D'},
    {'label': 'Uniform','crossover': 'Uniform'},
]
STYLES=['-','--',':']
for fn, dim, crossover in progressbar(list(itertools.product(
    FNS,
    DIMS,
    CROSSOVERS,
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
                {'config.alg_group.value': 'es_crossover'},
                {'config.device.value': 'cuda'},
                {'config.run_type.value': 'time,fitness'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.crossover.value': crossover['crossover']},
            ]
        }, keep_history=True)
        print()
        print(f"fn {fn}:{dim} for {crossover['label']}, runs: {len(runs)} for pop size {psize}")
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
    plt.title(f"{crossover['label']} crossover of $f_{{{fn}}}$ with {dim} dimensions")
    plt.savefig(f'runs/fitness_es_crossover_f{fn}_dim{dim}_{crossover["crossover"]}.pdf')
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
    f"runs/fitness_es_crossovers_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region ES crossover times
new_group('ES crossover time')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
FNS= [19, 24]
DIMS= [24, 128, 384]
CROSSOVERS = [
    {'label': 'Arithmetic','crossover': 'Arithmetic', 'c': 'tab:blue'},
    {'label': 'Blend','crossover': 'Blend', 'c': 'tab:orange'},
    {'label': 'One-point','crossover': 'OnePoint1D', 'c': 'tab:green'},
    {'label': 'Two-point','crossover': 'TwoPoint1D', 'c': 'tab:red'},
    {'label': 'Uniform','crossover': 'Uniform', 'c': 'tab:gray'},
]
TO_MEASURE = 'total_real_time'
for fn, dim in progressbar(list(itertools.product(
    FNS,
    DIMS,
))):
    plt.figure(figsize=FIGSIZE)
    maxval, minval = -math.inf, math.inf
    for crossover in CROSSOVERS:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': {'$in': POP_SIZES}},
                {'config.alg_group.value': 'es_crossover'},
                {'config.device.value': 'cpu'},
                {'config.run_type.value': 'time'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.crossover.value': crossover['crossover']},
            ]
        })
        print()
        print(f"CROSSOVER for fn {fn}:{dim} - {crossover['label']} has {len(runs)} runs")
        measure = np.zeros(len(POP_SIZES))
        run_count = np.zeros(len(POP_SIZES))
        for run in runs:
            try:
                s, c = run.summary, run.config
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
            c=crossover['c'],
            linestyle='-',
        )

        runs = MyRun.load_and_cache({
            "$and": [
                {'state': 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': {'$in': POP_SIZES}},
                {'config.alg_group.value': 'es_crossover'},
                {'config.device.value': 'cuda'},
                {'config.run_type.value': 'time,fitness'},
                {'config.bbob_fn.value': fn},
                {'config.bbob_dim.value': dim},
                {'config.es.crossover.value': crossover['crossover']},
            ]
        })
        print()
        print(f"CROSSOVER for fn {fn}:{dim} - {crossover['label']} has {len(runs)} runs")
        measure = np.zeros(len(POP_SIZES))
        run_count = np.zeros(len(POP_SIZES))
        for run in runs:
            try:
                s, c = run.summary, run.config
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
            c=crossover['c'],
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
    plt.savefig(f"runs/time_es_crossover_fn{fn}_{dim}d.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.axis('off')
legend = ax2.legend([
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0],[0],c=x['c']), CROSSOVERS)),
], [
    'CPU', 'CUDA',
    *list(map(lambda x: f'{x["label"]} crossover', CROSSOVERS))
], frameon=False, loc='lower center', ncol=10, )
fig2 = legend.figure
fig2.canvas.draw()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig(
    f"runs/time_es_crossover_legend.pdf",
    dpi="figure",
    bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
)
plt.close(fig2)
#endregion

#region ES mutation fitness
new_group('ES mutation fitness')
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
