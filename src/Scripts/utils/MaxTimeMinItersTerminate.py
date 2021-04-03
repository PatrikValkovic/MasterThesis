###############################
#
# Created by Patrik Valkovic
# 02.04.21
#
###############################
from typing import Tuple, Any, Dict
import time
import ffeat



class MaxTimeMinItersTerminate(ffeat.Pipe):
    DICT_INTERUPTED = 'time_interrupt'
    DICT_START = 'time_start'
    DICT_END = 'time_end'
    DICT_ITERS = 'time_iters'
    DICT_MAXMS = 'time_maxms'
    DICT_MINITERS = 'time_miniters'

    def __init__(self, max_ms: int, min_iters: int):
        self._maxms = max_ms
        self._min_iters = min_iters
        self._start = time.process_time()
        self._expected_end = self._start + self._maxms / 1000.0

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        assert 'ffeat_break' in kwargs
        assert 'iteration' in kwargs
        if time.process_time() > self._expected_end and kwargs['iteration'] >= self._min_iters:
            kwargs.update({
                MaxTimeMinItersTerminate.DICT_INTERUPTED: True,
                MaxTimeMinItersTerminate.DICT_START: self._start,
                MaxTimeMinItersTerminate.DICT_END: time.process_time(),
                MaxTimeMinItersTerminate.DICT_ITERS: kwargs['iteration'],
                MaxTimeMinItersTerminate.DICT_MAXMS: self._maxms,
                MaxTimeMinItersTerminate.DICT_MINITERS: self._min_iters,
            })
            kwargs['ffeat_break'](*args, **kwargs)
        return super().__call__(*args, **kwargs)
