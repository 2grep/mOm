# Optimizing ONEST

All times are, unless otherwise noted, tested using the `assisted.csv` dataset and calculated with 10 surfaces

## `match` Function
These were run single-threaded

| Version Name      | Time (s)          |
| ----------------- | ----------------- |
| early quit        | 68.40768909454346 |
| numpy all equals  | 920.9336943626404 |
| functools reduce  | 621.3175873756409 |
| itertools groupby | 629.38853931427   |
| all(first == x)    | 626.909182548523  |
| set length        | 630.9581050872803 |

- early quit
    ```python
    first = case[observers[0]]
    for observer in observers[1:len(observers)]:
        if case[observer] != first:
            return 0
    return 1
    ```

- numpy all equals
    ```python
    return int(np.all(case[observers] == case[observers[0]]))
    ```

- functools reduce
    ```python
    return functools.reduce(operator.eq, case[observers])
    ```

- itertools groupby
    ```python
    g = itertools.groupby(case[observers])
    return next(g, True) and not next(g, False)
    ```

- all(first == x)
    ```python
    iterator = iter(case[observers])
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)
    ```

- set length
    ```python
    return len(set(case[observers])) == 1
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
