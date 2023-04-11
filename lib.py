import os
from random import random
import numpy as np
from collections import deque
import typing as typ
import pandas as pd


def bucket(
        dataset: np.ndarray, 
        num_buckets: int, 
        range: typ.Union[typ.Tuple[int, int], list[int]] = (0, 1)) -> np.ndarray:
    '''
    Bucket dataset into num_buckets, assumes layer of dataset to bucket on is final 
    (i.e. if dataset.shape = (19, 240, 1000), will bucket into (19, 240, num_buckets)).

    range is exclusive on the right except the last bucket which is inclusive on both sides
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

def rotate(
        arr: np.ndarray, 
        num: int) -> np.ndarray:
    '''
    Rotate the ordering of the indices of arr
    '''
    ind = deque(np.arange(0, len(arr.shape)))
    ind.rotate(num)
    return np.transpose(arr, ind)


def data_reader(file_name):
    '''
    Read in data from file
    '''
    fname, fext = os.path.splitext(file_name)

    if fext == ".pkl":
        datasets_from_cache = True
        return pd.read_pickle(file_name)
    elif fext == ".npy":
        datasets_from_cache = True
        return np.load(file_name)
    else:
        data = pd.read_csv(file_name)
        print(data)
        return data

def random_unique_permutations(array, max_choices=-2):
    max_choices += 1
    prev_permutations = []
    while True:
        random.shuffle(array)
        new_permutation = array[:max_choices]
        while new_permutation in prev_permutations:
            random.shuffle(array)
            new_permutation = array[:max_choices]

        yield new_permutation
