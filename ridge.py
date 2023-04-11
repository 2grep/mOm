import numpy as np
import random as random
import typing as typ
import lib

def bucket(
        dataset: np.ndarray,
        num_buckets: int,
        range: typ.Union[typ.Tuple[int, int], list[int]] = (0, 1)
    ) -> np.ndarray:
    '''
    Bucket dataset into num_buckets, assumes layer of dataset to bucket on is final 
    (i.e. if `dataset.shape = (19, 240, 1000)`, will bucket into `(19, 240, num_buckets)`).

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

def opa_hist_ridge(
        treatment: np.ndarray,
        control: np.ndarray,
        certainty: int
    ) -> np.ndarray:
    '''
    Find "ridge" values for the treatment and control histograms
    (i.e. where `treatment >= control * certainty`).
    '''
    is_valid = treatment >= control * certainty
    # If the treatment has nothing, that may not be a fail, but it certainly isn't success
    # Increasing this flattening could help iron out "creeper" points in the ridge (it does) without being bad data handling (idk about that)
    # (e.g. treatment < 30 counts for statistical significance)
    is_valid[treatment == 0] = False
    return np.apply_along_axis(np.argmax, 1, is_valid)

def run_ridge(
        datasets: typ.Union[typ.Tuple[np.ndarray, np.ndarray], list[np.ndarray]],
        fname: str = "ridge",
        opa_slices: int = 10,
        certainty: float = 9
    ) -> None:
    '''
    Calculate "ridge" of treatment and control. In other words, find the number of cases 
    required for validation with an amount of certainty for each observer and opa.
    =====
    datasets: treatment, control
    '''

    # ranges = [obs_range(dataset, observer_slices) for dataset in datasets]
    treatment = lib.bucket(datasets[0], opa_slices)
    control = lib.bucket(datasets[1], opa_slices)
    print(control.shape)
    ridge = np.transpose(opa_hist_ridge(treatment, control, certainty))[::-1]

    np.savetxt(fname + ("" if ".csv" == fname[-4:] else ".csv"), ridge, fmt="%d", delimiter=",")

# TODO: refactor to calculate these datasets more simply - this requires simplifying onest_analysis.py
dataset_names = ["assisted_5case.npy", "unassisted_5case.npy"]
datasets = [np.transpose(lib.data_reader(set)) for set in dataset_names]

run_ridge(datasets, fname="ridge_5case")
