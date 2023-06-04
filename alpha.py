import matplotlib
import fitter
import multiprocessing as mp
import numpy as np
import scipy.stats as stats
import time
import typing
import matplotlib.pyplot as plt

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


def fit(data: np.ndarray) -> stats._distn_infrastructure.rv_continuous_frozen:
    try:
        params = stats.beta.fit(data)
        return stats.beta(*params)
    except Exception as e:
        print("Failed fit, returning `None`: ", e)
        return None

def compare(assisted, unassisted, alpha_error=.05):
    try:
        cutoff = unassisted.ppf(1 - alpha_error)
        beta_error = assisted.cdf(cutoff)
        return (cutoff, beta_error)
    except Exception as e:
        return None

# def main():
print("starting...")
# Find the OPA cutoff D for alpha_error probability to be greater than D
exp = "prostate_reader/"
group = "_5class"
root = "./data/"
results = "./results/" + exp + "/"
datasets = ["assisted", "unassisted"]
colors = ["red", "green"]
alpha_error = .05
histogram_bins = 30
res = {}

print("Fetching data...")
datasets = [np.load(root + exp + dataset + group + ".npy") for dataset in datasets]
datasets = np.transpose(np.asarray(datasets), (0, 3, 2, 1)) # (assisted/unassisted, observers, cases, surfaces)
print("Got datasets")

## * Fit each sample for observers x cases
print("Running fits...", flush=True)
# betas = np.apply_along_axis(fit, 3, datasets) # drops samples, gets params
# `test` is largely equivalent to the above line but lets me analyze the indices it fails on
def test() -> tuple[np.ndarray[stats._distn_infrastructure.rv_continuous_frozen], list[tuple[int]], int]:
    start = time.time()
    fits = np.empty_like(datasets[..., 0], dtype=stats._distn_infrastructure.rv_continuous_frozen)
    fails = []
    for i in range(datasets.shape[0]):
        for j in range(datasets.shape[1]):
            for k in range(datasets.shape[2]):
                try:
                    fits[i, j, k] = fit(datasets[i, j, k])
                except Exception as e:
                    print(f"Fail on ({i}, {j}, {k}): {e}")
                    fails.append((i, j, k))
    end = time.time()
    return (fits, fails, end - start)

res["test"] = test()
betas = res["test"][0]
print("Fits finished.")


## * Visual confirmation of emperical to theoretical
dims = (4, 6) # Want observers x cases
ratio = (2, 1)
scale = 1.4
obs = np.flip(np.linspace(
    datasets.shape[1] - 1, 0, 
    num=dims[0],
    endpoint=False,
    dtype=int
)) # obs should count up
cases = np.linspace(
    datasets.shape[2] - 1, 0,
    num=dims[1],
    endpoint=False,
    dtype=int
) # cases should count down (to match the top-left [0, 0] of axs)
fig, axs = plt.subplots(
    ncols=dims[0], 
    nrows=dims[1], 
    figsize=(ratio[0] * scale * dims[0], ratio[1] * scale * dims[1]),
    layout="constrained"
)
x = np.linspace(0, 1, num=500)
constant_indices = True
is_cdf = False
for row in range(axs.shape[0]):
    for col in range(axs.shape[1]):
        ind = (
            obs[col], 
            cases[row]
        )
        # Adjust indices to skip None in betas
        while any(
            [
                betas[g, *ind] == None 
                for g in range(datasets.shape[0])
            ]
        ):
            # increment cases if needed
            ind = (ind[0], ind[1] + 1)
            constant_indices = False
        ax = axs[row, col]
        ax.set_xlabel(ind[0] + 1)
        ax.set_ylabel(ind[1] + 1)
        for group in range(datasets.shape[0]):
            # Finally, graph emperical (hist) and theoretical (plot) pdfs for each SARAPE surface
            group_ind = (group, *ind)
            hist = ax.hist(
                datasets[*group_ind], 
                bins=histogram_bins, 
                range=(0, 1), 
                align="mid",
                density=True, # Draw normalized so area == 1
                cumulative=is_cdf, # Draw as CDF
                color=colors[group],
                alpha=.5
            )[0]
            if is_cdf:
                y = betas[*group_ind].cdf(x)
            else:
                y = betas[*group_ind].pdf(x)
            ax.plot(
                x, y,
                color=colors[group]
            )

# Clear out excess axes display elements
with np.nditer(
    axs,
    flags=[
        "multi_index",
        "refs_ok"
    ],
    op_flags=[
        "readwrite"
    ],
    op_dtypes=matplotlib.axes._axes.Axes
) as it:
    bottom_row = dims[1] - 1
    left_col = 0
    for ax in it:
        # For some god forsaken reason, ax is a zero-dimensional array which python just DOESN'T want to treat right
        # We use `ndarray.item()` to get the Python scalar value
        ax = ax.item()
        is_bottom_row = it.multi_index[0] == bottom_row
        is_left_col = it.multi_index[1] == left_col
        ax.tick_params(
            axis="both",
            which="both",
            top=False,
            bottom=is_bottom_row,
            left=False,
            right=False,
            labeltop=False,
            labelbottom=is_bottom_row,
            labelleft=False,
            labelright=False,
        )
        if constant_indices:
            if not is_left_col:
                ax.set_ylabel("")
            if not is_bottom_row:
                ax.set_xlabel("")

form = "CDF" if is_cdf else "PDF"
fig.suptitle(f"Various SARAPE Emperical vs. Theoretical {form}s")
fig.supxlabel("Observer Count")
fig.supylabel("Case Count")


## * Kolmogrov-Smirnov to check theoretical validity within p = .05
combo = np.concatenate((betas[..., np.newaxis], datasets), axis=-1)
def kstest(x):
    beta = x[0]
    if beta == None:
        return None
    data = x[1:].astype('f')
    return stats.kstest(data, beta.cdf)
res["kstest"] = np.apply_along_axis(kstest, 3, combo)


## * Beta comparison
print("Running cutoffs...", flush=True)
cutoff_and_theoretical_beta_error = np.transpose(
    np.apply_along_axis(
        lambda betas, alpha_error=.05: compare(*betas, alpha_error=alpha_error),
        0, 
        betas, 
        alpha_error=alpha_error
    ),
    (2, 1, 0)
)
t_beta = np.transpose(cutoff_and_theoretical_beta_error[..., 1])

# Getting emperical beta values
datasets = np.sort(datasets)
def _eppf(p, sorted_data):
    return sorted_data[int(p * len(sorted_data))]
eppf = np.vectorize(
    _eppf,
    signature="(),(n)->()"
)

def _ecdf(x, sorted_data):
    return np.searchsorted(sorted_data, x) / len(sorted_data)
ecdf = np.vectorize(
    _ecdf,
    signature="(),(n)->()"
)

assisted = datasets[0]
unassisted = datasets[1]
cutoff = eppf(1 - alpha_error, unassisted)
e_beta = ecdf(cutoff, assisted)
res["beta_diff"] = e_beta - t_beta




## * Save results
# print("saving...", flush=True)
# np.savetxt(results + "alpha_beta_cutoffs.csv", cutoff_and_beta_error[:, :, 0], delimiter=",", fmt="%.3f")
# np.savetxt(results + "alpha_beta_error.csv", cutoff_and_beta_error[:, :, 1], delimiter=",", fmt="%.3f")
plt.show()

# if __name__ == '__main__':
#     main()