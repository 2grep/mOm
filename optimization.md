# Optimizing ONEST

All times are, unless otherwise noted, tested using the `assisted.csv` dataset and calculated with 10 surfaces

## `match` Function
These were run single-threaded

### Pandas
I think this is really invalid because the `assisted.csv` file had the ground truth values which likely screwed things up.

| Version Name      | Time (s)          |
| ----------------- | ----------------- |
| early quit        | 68.40768909454346 |
| numpy all equals  | 920.9336943626404 |
| functools reduce  | 621.3175873756409 |
| itertools groupby | 629.38853931427   |
| all(first == x)    | 626.909182548523  |
| set length        | 630.9581050872803 |

### NumPy
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

### The Versions

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
    return np.all(match_list == match_list[0])
    ```

- functools reduce
    ```python
    return functools.reduce(operator.eq, match_list)
    ```

- itertools groupby
    ```python
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

## Multithreading
Using `early quite` for `match`. Multithreaded at the surface level (i.e. 10 seperate tasks generated). Single- and multi-threaded for 1000 surfaces used `unassisted.csv` and `assisted.csv` respectively so I could have each cached with 1000 surfaces.

| Level   | # of Surfaces | Threading    | Time (s)            |
| ------- | :-----------: | ------------ | :-----------------: |
| Surface | 10            | Singlethread | `69.70251798629761` |
| Surface | 10            | Max workers  | `68.77065896987915` |
| Surface | 1000          | Singlethread | `5709.312905073166` |
| Surface | 1000          | Max workers  | `6864.470486164093` |

What? Why is singlethread 1/3 of an hour *faster* than multithreading? Am I just bad at multithreading (probably)?
