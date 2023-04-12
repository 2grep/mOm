import numpy as np
import random as random
import typing as typ
import lib

def opa_hist_ridge(
        treatment: np.ndarray,
        control: np.ndarray,
        certainty: int
    ) -> np.ndarray:
    '''
    Find "ridge" values for the treatment and control histograms
    (i.e. where `treatment >= control * certainty`).
    '''
    is_valid = treatment > control * certainty
    # If the treatment has nothing, that may not be a fail, but it certainly isn't success
    # Increasing this flattening could help iron out "creeper" points in the ridge (it does) without being bad data handling (idk about that)
    # (e.g. treatment < 30 counts for statistical significance)
    is_valid[treatment == 0] = False
    return np.argmax(is_valid, axis=1)

def run_ridge(
        datasets: typ.Union[typ.Tuple[np.ndarray, np.ndarray], list[np.ndarray]],
        fname: str = "ridge",
        opa_slices: int = 10,
        certainty: float = 9
    ) -> None:
    '''
    Calculate "ridge" of treatment and control. In other words, find the number of cases 
    required for validation with an amount of certainty for each observer and opa.
    
    Parameters
    ----------
    datasets : treatment, control
    '''
    treatment = lib.bucket(datasets[0], opa_slices)
    control = lib.bucket(datasets[1], opa_slices)
    ridge = np.transpose(opa_hist_ridge(treatment, control, certainty))[::-1]

    np.savetxt(fname + ("" if ".csv" == fname[-4:] else ".csv"), ridge, fmt="%d", delimiter=",")

# TODO: refactor to calculate these datasets more simply - this requires simplifying onest_analysis.py
dataset_names = ["./data/assisted_2class.npy", "./data/unassisted_2class.npy"]
datasets = [np.transpose(lib.data_reader(set)) for set in dataset_names]

run_ridge(datasets, fname="./results/ridge_2class")
