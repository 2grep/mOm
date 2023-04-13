import numpy as np
import random as random
import typing as typ
import lib

def opa_hist_ridge(
        treatment: np.ndarray,
        control: np.ndarray,
        certainty: float = 9
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
        **kwargs
    ) -> np.ndarray:
    '''
    Calculate "ridge" of treatment and control. In other words, find the number of cases 
    required for validation with an amount of certainty for each observer and opa.
    
    Parameters
    ----------
    datasets : treatment, control
    fname : file name to save calculated ridge to as csv
        Will add `.csv` if not already present in `fname`.
    opa_slices : number of OPA buckets
        Dictates the number of buckets for `lib.bucket` to use.
    kwargs : keyword argyments to pass to `opa_hist_ridge`

    Returns
    -------
    ridge : calculated ridge as `(OPAs, observers)` with OPAs decreasing
    '''
    treatment = lib.bucket(datasets[0], opa_slices)
    control = lib.bucket(datasets[1], opa_slices)
    ridge = opa_hist_ridge(treatment, control, **kwargs)
    # Transpose for observers along the horizontal x-axis
    ridge = np.transpose(ridge)
    # Flip OPAs to be increasing going up the y-axis
    ridge = ridge[::-1]

    np.savetxt(fname + ("" if ".csv" == fname[-4:] else ".csv"), ridge, fmt="%d", delimiter=",")
    return ridge


# TODO: refactor to calculate these datasets more simply - this requires simplifying onest_analysis.py
dataset_names = ["./data/nottingham/nottingham.npy",
                 "./data/nottingham/tubulus.npy"]
datasets = [np.transpose(lib.data_reader(set)) for set in dataset_names]

run_ridge(datasets, fname="./results/nottingham/ridge")
