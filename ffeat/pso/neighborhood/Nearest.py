###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Union, Callable
import torch as t
from .Neighborhood import Neighborhood


class Nearest(Neighborhood):
    def __init__(self,
                 size: Union[int, Callable[..., int]],
                 norm: int = 2):
        self._size = self._handle_parameter(size)
        self._norm = norm

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        size = self._size(fitnesses, position, **kwargs)
        distances = t.subtract(position[:,None,:], position[None,:,:])
        distances = t.abs(distances, out=distances)
        distances = t.pow(distances, self._norm, out=distances)
        distances = t.sum(distances, dim=list(range(2, len(position.shape)+1)))
        min = t.argsort(distances, dim=-1)[:,:size]
        return min
