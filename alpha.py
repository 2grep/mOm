import fitter
import multiprocessing as mp
import numpy as np
import scipy.stats as stats
import time
import typing


# Apply along axis multiprocessing courtasy of https://stackoverflow.com/a/45555516/16755079
def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_apply_along_axis(
        func1d: typing.Callable,
        axis: int, 
        arr: np.ndarray, 
        *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs) for sub_arr in np.array_split(arr, mp.cpu_count())]

    pool = mp.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def fit(data):
    try:
        start = time.time()
        res = fitter.best_fit_distribution(data)
        end = time.time()
        print("Total:", end - start, flush=True)
        return res[0](*res[1])

    except:
        print("None")
        return None

def compare(assisted, unassisted, alpha_error=.05):
    cutoff = unassisted.ppf(1 - alpha_error)
    beta_error = assisted.cdf(cutoff)
    return (cutoff, beta_error)

def main():
    print("starting...")
    # Find the OPA cutoff D for alpha_error probability to be greater than D
    exp = "prostate_reader/"
    group = "_5class"
    root = "./data/"
    results = "./results/" + exp + "/"
    datasets = ["assisted", "unassisted"]
    alpha_error = .05

    print("Fetching data...")
    datasets = [np.load(root + exp + dataset + group + ".npy") for dataset in datasets]
    datasets = np.transpose(np.asarray(datasets), (0, 3, 2, 1))
    print("Got datasets")

    # Fit each sample for observers x cases
    print("Running fits...", flush=True)
    fits = parallel_apply_along_axis(fit, 3, datasets)  # drops samples, gets params
    print("Fits finished.")

    # Create beta distributions from fits
    # betas = np.apply_along_axis(lambda x: stats.beta(*x), 3, fits) # drops params, gets distributions
    betas = fits

    # Compare distribution to original data
    print("Comparing with KS test...", flush=True)
    to_compare = np.dstack([datasets, betas])
    # KstestResult broken into [statistic, pvalue] in axis 0
    dist_accuracy = np.apply_along_axis(lambda pair: stats.kstest(pair[1:].astype('f'), pair[0].cdf), 0, to_compare)

    # Compare assisted and unassisted distributions for a set Type I error to get the cutoff OPA for such error and resulting Type II error
    print("Running cutoffs...", flush=True)
    cutoff_and_beta_error = np.transpose(
        np.apply_along_axis(
            lambda betas, alpha_error=.05: compare(*betas, alpha_error=alpha_error),
            0, 
            betas, 
            alpha_error=alpha_error
        ), 
        (2, 1, 0)
    )

    # Save results
    print("saving...", flush=True)
    np.savetxt(results + "alpha_beta_cutoffs.csv", cutoff_and_beta_error[:, :, 0], delimiter=",", fmt="%.3f")
    np.savetxt(results + "alpha_beta_error.csv", cutoff_and_beta_error[:, :, 1], delimiter=",", fmt="%.3f")

if __name__ == '__main__':
    main()