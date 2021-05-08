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

CACHE_DIR = 'D:\\runscache'
api = wandb.Api()
FIGISZE_BIG=(12,8)  #(15,10)
FIGSIZE=(6, 4)  #(15,10)
SHOW_LEGEND = False

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
            runs = list(filter(lambda run: 'iteration' in run.summary, runs))
            with gzip.open(f'{CACHE_DIR}/{h.hexdigest()}.run', 'wb') as f:
                pickle.dump(runs, f)
        else:
            with gzip.open(f'{CACHE_DIR}/{h.hexdigest()}.run', 'rb') as f:
                runs = pickle.load(f)
        if reporting_metric:
            print(f"\n{h.hexdigest()}.run for\n{as_str}\nHas {len(runs)} runs")
        return runs
def new_group(name):
    #exit()
    print(f"RUNNING {name}")
def round_plotup(val, pos=0):
    base = math.floor(math.log10(val)) - pos
    round_to = 10 ** base
    return (int(val / round_to)+1) * round_to
def round_plotdown(val, pos=0):
    base = math.floor(math.log10(val)-pos if val > 0 else 0.0)
    round_to = 10 ** base
    return int(val / round_to) * round_to
def plot_generatelogticks(minval, maxval, ticks):
    lm = math.log(minval) if minval > 0 else 0.0
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


#region GA selection slowdown
new_group('GA selection slowdown')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384]
SELECTIONS = [
    {'label': 'Tournament selection', 'key': 'Tournament', 'c': 'tab:blue'},
    {'label': 'Roulette selection', 'key': 'Roulette', 'c': 'tab:orange'},
    {'label': 'SUS', 'key': 'StochasticUniversalSampling', 'c': 'tab:green'},
    {'label': 'Rank selection', 'key': 'Sequence', 'c': 'tab:red'},
]
TO_MEASURE = 'total_real_time'
plt.figure(figsize=FIGSIZE)
for selection in progressbar(SELECTIONS[:1]):
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
tournamentmeasure = measure
maxval, minval = -math.inf, math.inf
for selection in progressbar(SELECTIONS[1:]):
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
    measure = measure / tournamentmeasure
    minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
    plt.plot(
        POP_SIZES,
        measure,
        c=selection['c'],
        linestyle='-',
        label=selection['label']
    )
plt.title(f"Selection operators slowdown")
plt.xscale('log')
plt.yscale('log')
plt.xticks([32,128,512,2048,5000,16384])
plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim(32, 16384)
plt.xlabel('Literals')
minval = round_plotdown(minval)
maxval = round_plotup(maxval)
plt.ylim(minval, maxval)
plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
plt.minorticks_off()
plt.ylabel('Slowdown')
plt.legend()
plt.savefig(f"runs/time_ga_selection_slowdown.pdf")
plt.close()
#endregion

#region GA speedup
new_group('GA speed up')
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
plt.hlines(1, 32, 32768, colors='k', linestyle=':')
minval = math.inf
maxval = -math.inf
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
    cpu_measure = measure[run_count > 0] / run_count[run_count > 0]
    for pindex in np.where(run_count==0)[0]:
        print(f"\nWARNING GA ELITISM CPU fitness for {vars} literals with {POP_SIZES[pindex]} has no measurement")
        exit()
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
    cuda_measure = measure[run_count > 0] / run_count[run_count > 0]
    for pindex in np.where(run_count==0)[0]:
        print(f"\nWARNING GA GPU fitness for {vars} literals with {POP_SIZES[pindex]} has no measurement")
        exit()
    result = np.zeros_like(cpu_measure)
    is_better = cpu_measure > cuda_measure
    result[is_better] = (cpu_measure / cuda_measure)[is_better]
    is_worse = np.logical_not(is_better)
    result[is_worse] = -(cuda_measure / cpu_measure)[is_worse]
    plt.plot(
        POP_SIZES,
        cpu_measure / cuda_measure,
        c=VAR_C[str(vars)],
        linestyle='-',
        label=f'{vars} literals'
    )
    minval = min(minval, (cpu_measure / cuda_measure).min())
    maxval = max(maxval, (cpu_measure / cuda_measure).max())
plt.title(f"Genetic Algorithm speedup")
plt.xscale('log')
plt.xticks([32, 128, 512, 2048, 10240, 32768])
plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim(32, 32768)
plt.xlabel('Population size')
minval = round_plotdown(minval)
maxval = round_plotup(maxval)
plt.yscale('log')
plt.ylim(minval, maxval)
plt.yticks(plot_generatelogticks(minval, maxval, 7))
plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
plt.minorticks_off()
plt.ylabel('speedup')
plt.legend()
plt.savefig(f"runs/speedup_ga.pdf")
plt.close()
#endregion

#region running time C vs PyTorch
new_group('GA running time PyTorch vs C++')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
TO_MEASURE = 'total_real_time'
LEG_LINES = [
    mlines.Line2D([0],[0], linestyle='-', c='tab:blue'),
    mlines.Line2D([0],[0], linestyle='-', c='tab:orange'),
    mlines.Line2D([0], [0], linestyle='-', c='tab:green'),
    mlines.Line2D([0], [0], linestyle='-', c='tab:red'),
    mlines.Line2D([0], [0], linestyle='-', c='tab:gray'),
]
LEG_LABELS = [
    'C++ with 1 core',
    'C++ with 8 cores',
    'PyTorch with 1 core',
    'PyTorch with 8 cores',
    'PyTorch on GPU',
]
plt.figure(figsize=FIGISZE_BIG)
maxval, minval = -math.inf, math.inf
# C 1 core
for _ in range(1):
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.alg_group.value': 'ga_c'},
            {'config.run_type.value': 'time'},
            {'config.threads.value': 1},
            {'config.pop_size.value': {'$in': POP_SIZES}},
        ]
    })
    if len(runs) == 0:
        print("Warning, 0 runs for GA C implementation with 1 thread")
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
    if np.any(run_count == 0):
        print(f"\nWARNING, missing measurements for GA C with 1 thread for pop sizes {list(map(lambda x: POP_SIZES[x], np.where(run_count == 0)[0].tolist()))}")
    measure = measure[run_count > 0] / run_count[run_count > 0]
    minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c='tab:blue',
        linestyle='-',
    )
# C 8 core
for _ in range(1):
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.alg_group.value': 'ga_c'},
            {'config.run_type.value': 'time'},
            {'config.threads.value': 8},
            {'config.pop_size.value': {'$in': POP_SIZES}},
        ]
    })
    if len(runs) == 0:
        print("Warning, 0 runs for GA C implementation with 8 thread")
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
    if np.any(run_count == 0):
        print(f"\nWARNING, missing measurements for GA C with 8 thread for pop sizes {list(map(lambda x: POP_SIZES[x], np.where(run_count == 0)[0].tolist()))}")
    measure = measure[run_count > 0] / run_count[run_count > 0]
    minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c='tab:orange',
        linestyle='-',
    )
# PyTorch CPU 1 core
for _ in range(1):
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.alg_group.value': 'ga_1core'},
            {'config.run_type.value': 'time'},
            {'config.pop_size.value': {'$in': POP_SIZES}},
        ]
    })
    if len(runs) == 0:
        print("Warning, 0 runs for GA PyTorch CPU implementation with 1 thread")
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
    if np.any(run_count == 0):
        print(f"\nWARNING, missing measurements for GA PyTorch CPU with 1 thread for pop sizes {list(map(lambda x: POP_SIZES[x], np.where(run_count == 0)[0].tolist()))}")
    measure = measure[run_count > 0] / run_count[run_count > 0]
    minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c='tab:green',
        linestyle='-',
    )
# PyTorch CPU 8 core
for _ in range(1):
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.alg_group.value': 'ga_1'},
            {'config.run_type.value': 'time'},
            {'config.device.value': 'cpu'},
            {'config.sat.literals.value': 800},
            {'config.ga.elitism.value': False},
            {'config.pop_size.value': {'$in': POP_SIZES}},
        ]
    })
    if len(runs) == 0:
        print("Warning, 0 runs for GA PyTorch CPU implementation with 8 thread")
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
    if np.any(run_count == 0):
        print(f"\nWARNING, missing measurements for GA PyTorch CPU with 8 thread for pop sizes {list(map(lambda x: POP_SIZES[x], np.where(run_count == 0)[0].tolist()))}")
    measure = measure[run_count > 0] / run_count[run_count > 0]
    minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c='tab:red',
        linestyle='-',
    )
# PyTorch CUDA core
for _ in range(1):
    runs = MyRun.load_and_cache({
        "$and": [
            {'state': 'finished'},
            {'config.run_failed.value': False},
            {'config.alg_group.value': 'ga_1'},
            {'config.run_type.value': 'time,fitness'},
            {'config.device.value': 'cuda'},
            {'config.sat.literals.value': 800},
            {'config.ga.elitism.value': False},
            {'config.pop_size.value': {'$in': POP_SIZES}},
        ]
    })
    if len(runs) == 0:
        print("Warning, 0 runs for GA PyTorch CUDA implementation")
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
    if np.any(run_count == 0):
        print(f"\nWARNING, missing measurements for GA PyTorch CUDA for pop sizes {list(map(lambda x: POP_SIZES[x], np.where(run_count == 0)[0].tolist()))}")
    measure = measure[run_count > 0] / run_count[run_count > 0]
    minval, maxval = min(measure.min(), minval), max(measure.max(), maxval)
    plt.plot(
        np.array(POP_SIZES)[run_count > 0],
        measure,
        c='tab:gray',
        linestyle='-',
    )

plt.title(f"PyTorch versus C++ implementation")
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
if SHOW_LEGEND:
    plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
plt.savefig(f"runs/time_ga_c.pdf")
plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10, )
    fig2 = legend.figure
    fig2.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    fig2.savefig(
        f"runs/time_ga_c_legend.pdf",
        dpi="figure",
        bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    )
    plt.close(fig2)
#endregion

#region GA selection times
new_group('GA selection time')
NUM_Y_TICKS = 7
POP_SIZES = [32,128,200,512,1024,2048,5000,10240,16384]
SELECTIONS = [
    {'label': 'Tournament selection', 'key': 'Tournament', 'c': 'tab:blue'},
    {'label': 'Roulette selection', 'key': 'Roulette', 'c': 'tab:orange'},
    {'label': 'Stochastic Universal Sampling', 'key': 'StochasticUniversalSampling', 'c': 'tab:green'},
    {'label': 'Rank selection', 'key': 'Sequence', 'c': 'tab:red'},
]
TO_MEASURE = 'total_real_time'
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='k'),
    mlines.Line2D([0], [0], linestyle='--', c='k'),
    *(list(map(lambda s: mlines.Line2D([0], [0], linestyle='-', c=s['c']), SELECTIONS)))
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *(list(map(lambda s: s['label'], SELECTIONS)))
]
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
plt.xticks(POP_SIZES)
plt.gca().get_xaxis().set_major_formatter(mticker.ScalarFormatter())
plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim(32, 16384)
plt.xlabel('Literals')
minval = round_plotdown(minval)
maxval = round_plotup(maxval)
plt.ylim(minval, maxval)
plt.yticks(plot_generatelogticks(minval, maxval, NUM_Y_TICKS))
plt.gca().get_yaxis().set_major_formatter(mticker.ScalarFormatter(useOffset=False))
plt.minorticks_off()
plt.ylabel('Running time [s]')
if SHOW_LEGEND:
    plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
plt.savefig(f"runs/time_ga_selections.pdf")
plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10, )
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

#region GA normal and elitism fitness
new_group('GA fitness')
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
LEG_LINES = [
    mlines.Line2D([0],[0], linestyle='-', c='black'),
    mlines.Line2D([0],[0], linestyle='--', c='black'),
    mlines.Line2D([0],[0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x), PSIZE_C.values())),
]
LEG_LABELS = [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"Population size {x}", PSIZE_C.keys()))
]
for vars,elitism in progressbar(list(itertools.product(
    VARIABLES, ELITISM
))):
    plt.figure(figsize=FIGSIZE)
    maxval = -math.inf
    minval = 0
    iters_render = 1000
    for psize in POP_SIZES:
        runs = MyRun.load_and_cache({
            "$and": [
                {'state' : 'finished'},
                {'config.run_failed.value': False},
                {'config.pop_size.value': psize},
                {'config.alg_group.value': 'ga_1'},
                {'config.run_type.value': 'fitness'},
                {'config.ga.elitism.value': elitism},
                {'config.sat.literals.value': vars},
                {'createdAt': {'$gte': '2021-05-07T12:00:00'}},
            ]
        }, keep_history=True)
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
        #for r in runs:
        #    h = r.scan_history()
        #    tmp = []
        #    for s in h:
        #        tmp.append(s['fitness_q05'])
        #    plt.plot(range(len(tmp)), tmp, c=PSIZE_C[str(psize)], linestyle=STYLES[1], alpha=0.02)
        maxval = max(maxval, np.max(medians))
        minval = min(minval, np.max(best))
        #plt.plot(range(len(medians)), medians, c=PSIZE_C[str(psize)], linestyle=STYLES[0])
        plt.plot(range(len(q05)), q05,         c=PSIZE_C[str(psize)], linestyle=STYLES[1])
        #plt.plot(range(len(best)), best,       c=PSIZE_C[str(psize)], linestyle=STYLES[2])
    plt.xlim(0, iters_render)
    plt.xlabel('Generation')
    maxval = round_plotup(maxval, pos=1 if vars==2000 else 0)
    plt.ylim(minval, maxval)
    plt.ylabel('Objective function')
    plt.title(f"3SAT with {vars} literals")
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper right')
    plt.savefig(f'runs/fitness_ga{"_elitism" if elitism else ""}_3SAT_d{vars}.pdf')
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10, )
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
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x['c']), SCALES)),
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *list(map(lambda x: f'{x["label"]} scale', SCALES))
]
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
    plt.title(f"3SAT with {liters} literals")
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
    plt.savefig(f"runs/time_ga_scale_{liters}l.pdf")
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10, )
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
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x), VAR_C.values())),
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *list(map(lambda x: f'{x} literals', VAR_C.keys()))
]
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
plt.title(f"Running time of genetic algorithm with elitism")
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
if SHOW_LEGEND:
    plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
plt.savefig(f"runs/time_ga_elitism.pdf")
plt.close()

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
plt.title(f"Running time of genetic algorithm")
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
if SHOW_LEGEND:
    plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
plt.savefig(f"runs/time_ga.pdf")
plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10, )
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
NUM_Y_TICKS = 7
CLAUSES = [32,128,200,512,1024,2048,5000,10240,16384,32768]
TO_MEASURE = 'total_real_time'
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='tab:blue'),
    mlines.Line2D([0], [0], linestyle='-', c='tab:orange'),
]
LEG_LABELS = [
    'CPU', 'CUDA',
]
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
plt.title(f"In respect to number of clauses")
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
if SHOW_LEGEND:
    plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
plt.savefig(f"runs/time_ga_clausecount.pdf")
plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10, )
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
NUM_Y_TICKS = 7
VARS = [32,128,200,512,1024,2048,5000,10240,16384,32768]
TO_MEASURE = 'total_real_time'
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='tab:blue'),
    mlines.Line2D([0], [0], linestyle='-', c='tab:orange'),
]
LEG_LABELS = [
    'CPU', 'CUDA',
]
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
plt.title(f"In respect to number of literals")
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
if SHOW_LEGEND:
    plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
plt.savefig(f"runs/time_ga_varcount.pdf")
plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x), PSIZE_C.values())),
]
LEG_LABELS = [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"Population size {x}", PSIZE_C.keys()))
]
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
        #print(f"fn {fn}:{dim} for {crossover['label']}, runs: {len(runs)} for pop size {psize}")
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper right')
    plt.savefig(f'runs/fitness_es_crossover_f{fn}_dim{dim}_{crossover["crossover"]}.pdf')
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10, )
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
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x['c']), CROSSOVERS)),
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *list(map(lambda x: f'{x["label"]} crossover', CROSSOVERS))
]
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
        #print(f"CROSSOVER for fn {fn}:{dim} - {crossover['label']} has {len(runs)} runs")
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
        #print(f"CROSSOVER for fn {fn}:{dim} - {crossover['label']} has {len(runs)} runs")
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
    plt.savefig(f"runs/time_es_crossover_fn{fn}_{dim}d.pdf")
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
NAMES = {
    'AddFromNormal': 'Normal',
    'AddFromCauchy': 'Cauchy',
    'ReplaceUniform': 'Replace',
    'AdaptiveStep': 'Adaptive step',
}
PSIZE_C = {
    '32': 'tab:blue',
    '512': 'tab:orange',
    '10240': 'tab:green',
    '32768': 'tab:red',
}
STYLES=['-','--',':']
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x), PSIZE_C.values())),
]
LEG_LABELS = [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"Population size {x}", PSIZE_C.keys()))
]
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
        #print(f"fn {fn} with {dim} dim for {mut}, runs: {len(runs)} for pop size {psize}")
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
    plt.title(f"{NAMES[mut]} mutation of $f_{{{fn}}}$ with {dim} dimensions")
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper right')
    plt.savefig(f'runs/fitness_es_mutation_f{fn}_dim{dim}_{mut}.pdf')
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
    {'name': 'default', 'discard': False, 'replace': True},
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
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x), PSIZE_C.values())),
]
LEG_LABELS = [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"Population size {x}", PSIZE_C.keys()))
]
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
        #print(f"fn {fn} with {dim} dim for {schema['name']}, runs: {len(runs)} for pop size {psize}")
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
    plt.title(f"{schema['name']} mutation schema of BBOB $f_{{{fn}}}$ with {dim} dimensions")
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper right')
    plt.savefig(f'runs/fitness_es_schema_f{fn}_dim{dim}_{schema["name"]}.pdf')
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
    {'name': 'default', 'discard': False, 'replace': True, 'c': 'tab:blue'},
    {'name': 'comma', 'discard': True, 'replace': True, 'c': 'tab:orange'},
    {'name': 'plus', 'discard': False, 'replace': False, 'c': 'tab:green'},
]
TO_MEASURE = 'total_real_time'
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x['c']), SCHEMAS)),
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *list(map(lambda x: f'{x["name"]} schema', SCHEMAS))
]
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
        #print(f"SCHEMA for {fn} fn with {dim} dim, schema {schema['name']} has {len(runs)} runs")
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
        #print(f"SCHEMA for {fn} fn with {dim} dim, schema {schema['name']} has {len(runs)} runs")
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
    plt.savefig(f"runs/time_es_schema_fn{fn}_{dim}d.pdf")
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
NAMES = {
    'AddFromNormal': 'Normal mutation',
    'AddFromCauchy': 'Cauchy mutation',
    'ReplaceUniform': 'Replace mutation',
    'AdaptiveStep': 'Adaptive mutation',
}
TO_MEASURE = 'total_real_time'
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=MUT_COLORS[x]), MUTATIONS)),
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *[NAMES[mut] for mut in MUTATIONS]
]
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
    plt.savefig(f"runs/time_es_mutation_fn{fn}_{dim}d.pdf")
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
STYLES = ['-', '--', ':']
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle=':', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x), PSIZE_C.values())),
]
LEG_LABELS = [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    *list(map(lambda x: f"{x} particles", PSIZE_C.keys()))
]
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
        #print(neig)
        #print(f"psize {psize}, {fn}fn, runs: {len(runs)}")
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper right')
    plt.savefig(f'runs/fitness_pso_f{fn}_neigh{neig["label"].replace(" ","")}.pdf')
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *list(map(lambda x: mlines.Line2D([0], [0], c=x['color']), NEIGHBORHOODS)),
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *list(map(lambda x: x['label'], NEIGHBORHOODS))
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
    plt.savefig(f"runs/time_pso2006_fn{fn}_neigh.pdf")
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle=':', c='black'),
    mlines.Line2D([0], [0], c='tab:blue'),
    mlines.Line2D([0], [0], c='tab:orange'),
    mlines.Line2D([0], [0], c='tab:green'),
    mlines.Line2D([0], [0], c='tab:red'),
]
LEG_LABELS = [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    '32 particles', '512 particles',
    '10240 particles', '32768 particles',
]
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper right')
    plt.savefig(f'runs/fitness_pso2011_f{fn}.pdf')
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
STYLES = ['-','--',':']
NUM_Y_TICKS = 7
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    mlines.Line2D([0], [0], linestyle=':', c='black'),
    mlines.Line2D([0], [0], c='tab:blue'),
    mlines.Line2D([0], [0], c='tab:orange'),
    mlines.Line2D([0], [0], c='tab:green'),
    mlines.Line2D([0], [0], c='tab:red'),
]
LEG_LABELS = [
    'Fitness median', 'Fitness 0.05 quantile', 'Best fitness',
    '32 particles', '512 particles',
    '10240 particles', '32768 particles',
]
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper right')
    plt.savefig(f'runs/fitness_pso2006_f{fn}.pdf')
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
    fig2 = legend.figure
    fig2.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    fig2.savefig(
        f"runs/fitness_pso2006_legend.pdf",
        dpi="figure",
        bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    )
    plt.close(fig2)
#endregion

#region PSO2006 run times
new_group('PSO2006 run times')
NUM_Y_TICKS = 7
POP_RENDER = [32, 128, 512, 2048, 10240, 32768]
pop_list = [32, 128, 200, 512, 1024, 2048, 5000, 10240, 16384, 32768]
DIM_COLORS = {
    '6': 'tab:blue',
    '32': 'tab:orange',
    '128': 'tab:green',
    '384': 'tab:red',
}
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *[mlines.Line2D([0], [0], c=col) for col in DIM_COLORS.values()]
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *[f'Problem dimension {dom}' for dom in DIM_COLORS.keys()]
]
for fn, in progressbar(list(itertools.product(
        [1, 7, 15, 19, 22, 24],
))):
    plt.figure(figsize=FIGSIZE)
    maxval = -math.inf
    minval = math.inf
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
                progress = s['iteration'] / s['max_iteration']
                i = pop_list.index(c['pop_size'])
                measure[i] += s['total_perf_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        minval = min(minval, np.min(measure / run_count))
        maxval = max(maxval, np.max(measure / run_count))
        plt.plot(pop_list, measure / run_count,
                 linestyle='-', c=DIM_COLORS[str(dim)])
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
                progress = s['iteration'] / s['max_iteration']
                psize = c['pop_size']
                i = pop_list.index(c['pop_size'])
                measure[i] += s['total_perf_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        minval = min(minval, np.min(measure / run_count))
        maxval = max(maxval, np.max(measure / run_count))
        plt.plot(pop_list, measure / run_count,
                 linestyle='--', c=DIM_COLORS[str(dim)])
    plt.title(f"BBOB function $f_{{{fn}}}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(POP_RENDER)
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
    plt.savefig(f"runs/time_pso2006_fn{fn}_alldim.pdf")
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
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
NUM_Y_TICKS = 7
DIM_COLORS = {
    '6': 'tab:blue',
    '32': 'tab:orange',
    '128': 'tab:green',
    '384': 'tab:red',
}
LEG_LINES = [
    mlines.Line2D([0], [0], linestyle='-', c='black'),
    mlines.Line2D([0], [0], linestyle='--', c='black'),
    *[mlines.Line2D([0], [0], c=col) for col in DIM_COLORS.values()]
]
LEG_LABELS = [
    'CPU', 'CUDA',
    *[f'Problem dimension {dom}' for dom in DIM_COLORS.keys()]
]
for fn, in progressbar(list(itertools.product(
        [1, 7, 15, 19, 22, 24],
))):
    plt.figure(figsize=FIGSIZE)
    maxval = -math.inf
    minval = math.inf
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
                progress = s['iteration'] / s['max_iteration']
                i = pop_list.index(c['pop_size'])
                measure[i] += s['total_perf_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        minval = min(minval, np.min(measure / run_count))
        maxval = max(maxval, np.max(measure / run_count))
        plt.plot(pop_list, measure / run_count,
                 linestyle='-', c=DIM_COLORS[str(dim)])
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
                progress = s['iteration'] / s['max_iteration']
                psize = c['pop_size']
                i = pop_list.index(c['pop_size'])
                measure[i] += s['total_perf_time'] / progress
                run_count[i] += 1
            except:
                traceback.print_exc()
                print(run)
        minval = min(minval, np.min(measure / run_count))
        maxval = max(maxval, np.max(measure / run_count))
        plt.plot(pop_list, measure / run_count,
                 linestyle='--', c=DIM_COLORS[str(dim)])
    plt.title(f"BBOB function $f_{{{fn}}}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(POP_RENDER)
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
    if SHOW_LEGEND:
        plt.legend(LEG_LINES, LEG_LABELS, loc='upper left')
    plt.savefig(f"runs/time_pso2011_fn{fn}_alldim.pdf")
    plt.close()

if not SHOW_LEGEND:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(LEG_LINES, LEG_LABELS, frameon=False, loc='lower center', ncol=10)
    fig2 = legend.figure
    fig2.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    fig2.savefig(
        f"runs/time_pso2011_alldim_legend.pdf",
        dpi="figure",
        bbox_inches=legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    )
#endregion
