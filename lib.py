import math
import os
import numpy.random as random
import numpy as np
from collections import deque
import typing as typ
import time

## Type Variables ##
T = typ.TypeVar('T')

## Functions ##

def bucket(
        dataset: np.ndarray,
        num_buckets: int,
        range: typ.Union[tuple[int, int], list[int]] = (0, 1)
    ) -> np.ndarray:
    '''
    Bucket dataset into num_buckets, assumes layer of dataset to bucket on is final 
    (i.e. if dataset.shape = (19, 240, 1000), will bucket into (19, 240, num_buckets)).
    Basically just numpy.histogram but makes sure the bins are always the same

    range is exclusive on the right except the last bucket which is inclusive on both sides.
    '''
    spacer = (range[1] - range[0]) / num_buckets
    return np.apply_along_axis(
        lambda x: np.histogram(
            x, 
            bins=np.arange(
                range[0], 
                range[1] + spacer, 
                spacer
            )
        )[0], # histogram returns a tuple, the first item is the histogram array
        len(dataset.shape) - 1, # apply along the last axis
        dataset
    )

def data_reader(
        fname: str,
        names: list[str] = [],
        exts: list[str] = []
    ) -> np.ndarray:
    '''
    Read in data from file.
    Appends file names and extensions to names and exts repectively if specified.
    '''
    _fname, fext = os.path.splitext(fname)
    names.append(_fname)
    exts.append(fext)
    
    if fext == ".npy":
        return np.load(fname)
    
    elif fext == ".csv":
        data = np.loadtxt(fname, delimiter=",")
        return data
    raise Exception(f"{fext} is an unrecognized file extension for `fname`.")

def random_unique_permutations(
    arr: np.ndarray,
    call_count: int = None
) -> typ.Generator[np.ndarray, None, None]:
    '''
    Generate random, unique (i.e. will never return sequences with the exact same order) 
    permutations of arr.

    Parameters
    ----------
    arr : array to permute
    call_count : number of times `next` will be called
        If supplied, improves performance by not having to check with every call 
        but may enter infite loop if `next` is called more than `call_count` times.

    Yields
    ------
    random_unique_permutations : a new random, unique permutation of seq 

    Raises
    ------
    AssertException : `call_count` is greater than `len(arr)` factorial 
        (i.e. will call more than the number of possible permutations).
    AssertException : `next` called more times than `len(arr)!` 
        (`call_count == None`)
    '''
    rng = random.default_rng()
    prev = set()
    prev_add = prev.add
    max = math.factorial(len(arr))

    def next(rng, arr):
        rng.shuffle(arr)

        hasharr = arr.data.tobytes()
        while hasharr in prev:
            rng.shuffle(arr)
            hasharr = arr.data.tobytes()
        prev_add(hasharr)

        return arr
    
    if call_count != None:
        assert call_count <= max, "Cannot generate more unique permutations than exist"
        while True:
            yield next(rng, arr)

    else:
        while True:
            assert len(prev) != max, "Cannot get more unique permutations than exist"

            yield next(rng, arr)

def all_match(match_list: typ.Iterable) -> bool:
    '''
    Check if all in `match_list` are the same value
    '''
    first = match_list[0]
    for item in match_list[1:]:
        if item != first:
            return False
    return True
