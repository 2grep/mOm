import numpy as np
import random as random
import typing as typ
import scipy.stats as stats
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

def opa_ks_ridge(
        treatment: np.ndarray,
        control: np.ndarray,
        certainty: float = .95
    ) -> np.ndarray:
    '''
    Use the two-sample Kolmogorov-Smirnov test to identify the ridge of cases

    Parameters
    ----------
    certainty in [0, 1] : level of certainty that treatment and control are resolvable
    '''
    assert 0 <= certainty and certainty <= 1, "Certainty must be between 0 and 1 inclusive."
    assert treatment.shape[:-1] == control.shape[:-1], "Non-sample dimensions of treatment and control must be the same"


    null_pvalues = np.empty(treatment.shape[:-1])
    print(np.amax(control))
    print(f"Running with {treatment.shape[0]} at base")
    for obs in range(treatment.shape[0]):
        print("Base:", obs)
        null_pvalues[obs] = ks_flat(treatment[obs], control[obs])
        # for i, (x, y) in enumerate(zip(treatment[obs], control[obs])):
        #     null_pvalues[obs, i] = stats.ks_2samp(x, y, alternative="two-sided", mode="exact").pvalue
    # null_threshold = 1 - certainty
    # is_valid = null_pvalues <= null_threshold

    return null_pvalues

def ks_flat(treatment, control):
    '''
    Check Kolmogrov-Smirnov 2-sided test for treatment and control
    
    Assumes `treatment` and `control` are of the same size
    '''
    pvalues = np.empty(treatment.shape[:-1])
    for i, (x, y) in enumerate(zip(treatment, control)):
        pvalues[i] = stats.ks_2samp(x, y, alternative="two-sided", mode="auto").pvalue
    return pvalues

def run_ks_ridge(
        datasets,
        fname = "ridge",
        **kwargs
    ):
    ridge = opa_ks_ridge(datasets[0], datasets[1], **kwargs)
    np.savetxt(fname + ("" if ".csv" == fname[-4:] else ".csv"), ridge, fmt="%d", delimiter=",")


def run_hist_ridge(
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
dataset_names = ["./data/prostate_reader/assisted_5class.npy",
                 "./data/prostate_reader/unassisted_5class.npy"]
datasets = [np.transpose(lib.data_reader(set)) for set in dataset_names]

run_hist_ridge(datasets, fname="./results/ridge", opa_slices=10)
# opa_ks_ridge(datasets[0], datasets[1])
# run_ks_ridge(datasets, fname="ridge_greater", certainty=.95)
