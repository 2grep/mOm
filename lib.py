import os
from numpy import typing as nptyp
import numpy as np
from collections import deque
import typing as typ

import pandas as pd


def bucket(dataset: nptyp.ArrayLike, 
           num_buckets: int, 
           range: typ.Union[typ.Tuple[int, int], list[int]] = (0, 1)) -> nptyp.NDArray:
    '''
    Bucket dataset into num_buckets, assumes layer of dataset to bucket on is final 
    (i.e. if dataset.shape = (19, 240, 1000), will bucket into (19, 240, num_buckets)).

    range is exclusive on the right
    '''
    # dataset = np.transpose(dataset, deque(dataset.shape).rotate(1))

    # num_buckets in the range of [ range[0], range[1] )
    buckets = np.arange(range[0], range[1], 1/num_buckets).astype("float64", copy=False)
    # replace values of dataset with appropriate bin in buckets
    digitized = np.digitize(dataset, buckets) - 1

    # bincount counts the number of times each integer occurs in a 1d-array
    # in other words, we are replacing each layer with the count of the frequency of each bucket in that layer
    # (observers, cases, OPA) 
    # (e.g. bucket_reduced[5-2][100-1][int(.5*100)] is the number of points at 5 observers, 100 cases, and an OPA of .5)
    return np.apply_along_axis(lambda x: np.bincount(x, minlength=100), len(digitized.shape) - 1, digitized)

def rotate(arr, num):
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
