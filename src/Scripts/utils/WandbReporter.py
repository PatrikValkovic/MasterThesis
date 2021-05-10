###############################
#
# Created by Patrik Valkovic
# 01.04.21
#
###############################
import wandb
import time


class WandbReporter:
    def __init__(self, wandbinit = None):
        self._wandbinit = wandbinit or {}

    def __enter__(self):
        self._run = wandb.init(project='thesis', allow_val_change=True, **self._wandbinit)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._run.finish()

    @property
    def run(self):
        return self._run

    def __call__(self, *args, **kwargs):
        d = dict(kwargs)
        if 'orig_fitness' in d:
            del d['orig_fitness']
        if 'new_fitness' in d:
            del d['new_fitness']
        wandb.log(d, step=kwargs['iteration'])
        return args, kwargs


class WandbExecutionTime(WandbReporter):
    def __enter__(self):
        super().__enter__()
        self._start_proc = time.process_time()
        self._last_iter_proc = self._start_proc
        self._start_perf = time.perf_counter()
        self._last_iter_perf = self._start_perf
        self._start_time = time.time()
        self._last_iter_time = self._start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, *args, **kwargs):
        now_proc, now_perf, now_time = time.process_time(), time.perf_counter(), time.time()
        wandb.log({
            'iteration_proc_time': now_proc - self._last_iter_proc,
            'total_proc_time': now_proc - self._start_proc,
            'iteration_perf_time': now_perf - self._last_iter_perf,
            'total_perf_time': now_perf - self._start_perf,
            'iteration_real_time': now_time - self._last_iter_time,
            'total_real_time': now_time - self._start_time,
        }, step=kwargs['iteration'])
        self._last_iter_proc, self._last_iter_perf, self._last_iter_time = now_proc, now_perf, now_time
        return super().__call__(*args, **kwargs)
