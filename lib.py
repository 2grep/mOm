import os
import random
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
        )[0],
        len(dataset.shape) - 1,
        dataset
    )

# TODO: I think this is deprecated. If so, get rid of it; if not, compare deque to list for rotating
def rotate(
        arr: np.ndarray, 
        num: int
    ) -> np.ndarray:
    '''
    Rotate the ordering of the indices of arr
    '''
    ind = deque(np.arange(0, len(arr.shape)))
    ind.rotate(num)
    return np.transpose(arr, ind)

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
            arr: typ.MutableSequence[typ.Type[T]]
    ) -> typ.Generator[typ.MutableSequence[typ.Type[T]], None, None]:
    '''
    Generate random, unique permutations of arr upto max_choices number of values.

    Effectively computes `len(seq)` choose `max_choices` permutations.

    Parameters
    ----------
    seq : sequence to permute
    max_choices : maximum number of choices to pull from each permutation

    Yields
    ------
    random_unique_permutations : a new random, unique permutation of seq 
    '''
    # TODO: check that this really is unique
    while True:
        random.seed(time.time())
        random.shuffle(arr)
        yield arr

def match(match_list: typ.Iterable) -> bool:
    '''
    Check if all in `match_list` are the same value
    '''
    first = match_list[0]
    for item in match_list[1:]:
        if item != first:
            return False
    return True
