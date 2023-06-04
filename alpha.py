import matplotlib
import numpy as np
import scipy.stats as stats
import time
import matplotlib.pyplot as plt

def fit(data: np.ndarray) -> stats.rv_continuous:
    try:
        params = stats.beta.fit(data)
        return stats.beta(*params)
    except Exception as e:
        print("Failed fit, returning `None`: ", e)
        return None

def compare(
    assisted: np.ndarray[stats.rv_continuous], 
    unassisted: np.ndarray[stats.rv_continuous], 
    alpha_error: float = .05
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    try:
        cutoff = unassisted.ppf(1 - alpha_error)
        beta_error = assisted.cdf(cutoff)
        return (cutoff, beta_error)
    except Exception as e:
        return None

# `test` is largely equivalent to the above line but lets me analyze the indices it fails on
def test() -> tuple[np.ndarray[stats.rv_continuous], list[tuple[int]], float]:
    start = time.time()
    fits = np.empty_like(datasets[..., 0], dtype=stats.rv_continuous)
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
# res["test"] = test()
# betas = res["test"][0]
betas = np.apply_along_axis(fit, -1, datasets)
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
    for ax in it:
        ax = ax.item()

        ind = (
            obs[it.multi_index[1]], 
            cases[it.multi_index[0]]
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
        ax.set_xlabel(ind[0] + 1)
        ax.set_ylabel(ind[1] + 1)
        for group in range(datasets.shape[0]):
            # Finally, graph emperical (hist) and theoretical (plot) pdfs for each SARAPE surface
            group_ind = (group, *ind)
            hist = ax.hist(
                datasets[group_ind[0], group_ind[1], group_ind[2]], 
                bins=histogram_bins, 
                range=(0, 1), 
                align="mid",
                density=True, # Draw normalized so area == 1
                cumulative=is_cdf, # Draw as CDF
                color=colors[group],
                alpha=.5
            )[0]
            if is_cdf:
                y = betas[group_ind[0], group_ind[1], group_ind[2]].cdf(x)
            else:
                y = betas[group_ind[0], group_ind[1], group_ind[2]].pdf(x)
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

graph_form = "CDF" if is_cdf else "PDF"
fig.suptitle(f"Various SARAPE Emperical vs. Theoretical {graph_form}s")
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
res["kstest"] = np.apply_along_axis(kstest, -1, combo)


## * Beta comparison
print("Running cutoffs...", flush=True)
res["theoretical"] = np.transpose(
    np.apply_along_axis(
        lambda betas, alpha_error=.05: compare(*betas, alpha_error=alpha_error),
        0, 
        betas, 
        alpha_error=alpha_error
    ),
    (2, 1, 0)
)
t_beta = np.transpose(res["theoretical"][..., 1])

# Getting emperical beta values
datasets = np.sort(datasets)
def _eppf(
    p: float, 
    sorted_data: list[float]
) -> float:
    return sorted_data[int(p * len(sorted_data))]
eppf = np.vectorize(
    _eppf,
    signature="(),(n)->()"
)

def _ecdf(
    x: float, 
    sorted_data: list[float]
) -> float:
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
print("saving...", flush=True)
np.savetxt(results + "theoretical_cutoffs.csv", res["theoretical"][:, :, 0], delimiter=",", fmt="%.3f")
np.savetxt(results + "theoretical_beta.csv", res["theoretical"][:, :, 1], delimiter=",", fmt="%.3f")
np.savetxt(results + "e-t_beta_diff.csv", res["beta_diff"], delimiter=",", fmt="%.3f")
plt.savefig(
    f"{results}/{graph_form}.png",
    bbox_inches="tight",
    transparent=False,
    dpi=1000
)
plt.show()

# if __name__ == '__main__':
#     main()