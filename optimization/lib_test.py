import numpy.random
import math
from timeit import timeit
import numpy as np

def permute(seq):
    rng = numpy.random.default_rng()
    while True:
        rng.shuffle(seq)
        yield seq

def checked_permute(
        seq: np.ndarray, 
        count: int = None
    ):
    rng = numpy.random.default_rng()
    prev = set()
    prev_add = prev.add
    max = math.factorial(len(seq))
    if count != None:
        if count >= max:
            raise Exception("Cannot get more unique permutations than exist")
        while True:
            rng.shuffle(seq)

            hashseq = seq.data.tobytes()
            while hashseq in prev:
                rng.shuffle(seq)
                hashseq = seq.data.tobytes()
            prev_add(hashseq)

            yield seq
    else:
        while True:
            assert len(prev) != max, "Cannot get more unique permutations than exist"
            
            rng.shuffle(seq)

            hashseq = seq.data.tobytes()
            while hashseq in prev:
                rng.shuffle(seq)
                hashseq = seq.data.tobytes()
            prev_add(hashseq)

            yield seq
gen = checked_permute(np.arange(0, 20))
print(timeit(lambda: next(gen), number=1000000))
gen = permute(np.arange(0, 20))
print(timeit(lambda: next(gen), number=1000000))