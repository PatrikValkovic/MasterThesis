###############################
#
# Created by Patrik Valkovic
# 01.04.21
#
###############################
import wandb

class WandbReporter:
    def __init__(self, wandbinit = None):
        self._wandbinit = wandbinit or {}

    def __enter__(self):
        self._run = wandb.init(project='thesis', **self._wandbinit)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._run.finish()

    @property
    def run(self):
        return self._run

    def __call__(self, *args, **kwargs):
        wandb.log(kwargs, step=kwargs['iteration'])
        return args, kwargs
