# Optimizing ONEST

All times are, unless otherwise noted, tested using the `assisted.csv` dataset unless otherwise noted.

## `match` Function
These were run single-threaded

## Pandas
I think this is really invalid because the `assisted.csv` file had the ground truth values which likely screwed things up.

| Version Name      | Time (s)          |
| ----------------- | ----------------- |
| early quit        | 68.40768909454346 |
| numpy all equals  | 920.9336943626404 |
| functools reduce  | 621.3175873756409 |
| itertools groupby | 629.38853931427   |
| all(first == x)    | 626.909182548523  |
| set length        | 630.9581050872803 |

## NumPy
For 10 surfaces:
| Version Name      | Time (s)           |
| ----------------- | ------------------ |
| early quit        | 6.089831829071045  |
| numpy all equals  | 15.421066045761108 |
| functools reduce  | 44.58276987075806  |
| itertools groupby | 6.2754082679748535 |
| all(first == x)    | 7.123004913330078  |
| set length        | 7.707198143005371  |

For 100 surfaces:
| Version Name      | Time (s)          |
| ----------------- | ----------------- |
| early quit        | 61.479896068573   |
| itertools groupby | 61.39276599884033 |

For 1000 surfaces:
| Version Name      | Time (s)          |
| ----------------- | ----------------- |
| early quit        | 596.1405961513519 |
| itertools groupby | 612.1813862323761 |

## The Versions

- early quit
    ```python
    first = match_list[0]
    for item in match_list[1:]:
        if item != first:
            return False
    return True
    ```

- numpy all equals
    ```python
    import numpy as np
    ------------------
    return np.all(match_list == match_list[0])
    ```

- functools reduce
    ```python
    import fuctools, operator
    ----------------
    return functools.reduce(operator.eq, match_list)
    ```

- itertools groupby
    ```python
    import functools
    ----------------
    g = itertools.groupby(match_list)
    return next(g, True) and not next(g, False)
    ```

- all(first == x)
    ```python
    iterator = iter(match_list)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)
    ```

- set length
    ```python
    return len(set(match_list)) == 1
    ```

# `random_unique_permutations` Function

These are all computed with 20 observers in the ONEST method and the early quit method for `match`

### Timing

| #  | # of Curves | Method                                   | User Time  | Fails |
| -- | ----------- | ---------------------------------------- | ---------- | ----- |
| 1  | 1,000       | `random` unchecked                       | 0m6.407s   | N/A   |
| 2  | 1,000       | `random` with `in` check                 | 0m7.005s   | 0     |
| 3  | 1,000       | `numpy.random.Generator` unchecked       | 0m6.116s   | N/A   |
| 4  | 1,000       | `numpy.random.Generator` with `in` check | 0m6.447s   | 0     |
| 5  | 10,000      | `random` unchecked                       | 0m50.399s  | N/A   |
| 6  | 10,000      | `random` with `in` check                 | 0m51.152s  | 0     |
| 7  | 10,000      | `numpy.random.Generator` unchecked       | 0m49.310s  | N/A   |
| 8  | 10,000      | `numpy.random.Generator` with `in` check | 0m52.156s  | 0     |
| 9  | 100,000     | `random` unchecked                       | 8m9.004s   | N/A   |
| 10 | 100,000     | `random` with `in` check                 | 13m25.277s | 0     |
| 11 | 100,000     | `numpy.random.Generator` unchecked       | 8m5.589s   | N/A   |
| 12 | 100,000     | `numpy.random.Generator` with `in` check | 13m19.908s | 0     |

(9) to (10) is a 64.7% increase in time.
(9) to (11) is a 0.69% decrease in time.
(10) to (12) is a 0.67% decrease in time.
(11) to (12) is 64.7% inccrease in time

### Counting Fails

These tests are run with the `numpy.random.Generator` with `in` check method and 10000 curves. First fail is indexed starting at 1.

| # of Observers | First Fail | # of Fails | % Fail |
| -------------- | ---------- | ---------- | ------ |
| 1              | 2          | 9999       | 99.99% |
| 2              | 3          | 9998       | 99.98% |
| 3              | 4          | 9994       | 99.9%  |
| 4              | 5          | 9976       | 99.8%  |
| 5              | 19         | 9880       | 98.8%  |
| 6              | 34         | 9280       | 92.8%  |
| 7              | 148        | 5664       | 56.6%  |
| 8              | 650        | 1157       | 11.6%  |
| 9              | 1049       | 140        | 14%    |
| 10             | 3010       | 14         | 1.4%   |
| 11             | 5145       | 2          | .02%   |
| 12             | N/A        | 0          | 0%     |
| 13             | N/A        | 0          | 0%     |
| 14             | N/A        | 0          | 0%     |
| 15             | N/A        | 0          | 0%     |
| 16             | N/A        | 0          | 0%     |
| 17             | N/A        | 0          | 0%     |
| 18             | N/A        | 0          | 0%     |
| 19             | N/A        | 0          | 0%     |
| 20             | N/A        | 0          | 0%     |

## The Methods

- `random` unchecked
    ```python
    import random, time
    --------------------
    while True:
        random.seed(time.time())
        random.shuffle(arr)
        yield arr
    ```

- `random` with `in` check:
    ```python
    import random, time
    fail = []
    --------------------
    prev = []
    while True:
        random.seed(time.time())
        random.shuffle(arr)

        arrlist = list(arr)
        if arrlist in prev:
            fail.append(1)
        prev.append(arrlist)

        yield arr
    ```

- `numpy.random.Generator` unchecked
    ```python
    import numpy.random as random
    ------------------------------
    rng = random.default_rng()
    while True:
        rng.shuffle(arr)
        yield arr
    ```

- `numpy.random.Generator` with `in` check
    ```python
    import numpy.random as random
    fail = []
    ------------------------------
    rng = random.default_rng()
    prev = []
    while True:
        rng.shuffle(arr)

        arrlist = list(arr)
        if arrlist in prev:
            fail.append(1)
        prev.append(arrlist)

        yield arr
    ```

# Multithreading
Using `early quite` for `match`. Multithreaded at the surface level (i.e. 10 seperate tasks generated). Single- and multi-threaded for 1000 surfaces used `unassisted.csv` and `assisted.csv` respectively so I could have each cached with 1000 surfaces.

| Level   | # of Surfaces | Threading    | Time (s)            |
| ------- | :-----------: | ------------ | :-----------------: |
| Surface | 10            | Singlethread | `69.70251798629761` |
| Surface | 10            | Max workers  | `68.77065896987915` |
| Surface | 1000          | Singlethread | `5709.312905073166` |
| Surface | 1000          | Max workers  | `6864.470486164093` |

What? Why is singlethread 1/3 of an hour *faster* than multithreading? Am I just bad at multithreading (probably)?
