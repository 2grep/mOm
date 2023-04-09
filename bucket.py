import numpy as np
from collections import deque

# (unique surfaces, cases, observers - 1)
dataset = np.load("assisted.npy")
dataset = np.transpose(dataset, deque(dataset.shape).rotate(1))

num_buckets = 100
buckets = np.arange(0.0, 1, 1/num_buckets)
digitized = np.digitize(dataset, buckets) - 1

# (observers, cases, OPA) 
# (e.g. bucket_reduced[5-2][100-1][int(.5*100)] is the number of points at 5 observers, 100 cases, and an OPA of .5)
bucket_reduced = np.apply_along_axis(lambda x: np.bincount(x, minlength=100), 2, digitized)