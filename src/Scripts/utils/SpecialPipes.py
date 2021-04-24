###############################
#
# Created by Patrik Valkovic
# 4/4/2021
#
###############################
from typing import Tuple, Any, Dict
import ffeat
from ffeat import Pipe


class FSubtractPipe(Pipe):
    def __init__(self, val):
        self._val = val
    def __call__(self, fitnesses, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return (fitnesses - self._val, *args), kwargs


class FSubtractFromPipe(FSubtractPipe):
    def __call__(self, fitnesses, *args, **kwargs):
        return (self._val - fitnesses, *args), kwargs


class DiscardValsPipe(Pipe):
    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return tuple(), kwargs


class SidewayPipe(Pipe):
    def __init__(self, *steps):
        self._flow = ffeat.flow.Parallel(
            ffeat.Pipe(),
            ffeat.flow.Sequence(
                *steps,
                DiscardValsPipe()
            )
        )

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return self._flow(*args, **kwargs)
