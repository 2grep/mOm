import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import norm
import random as random
import lib
import typing as typ

# * This is basically deprecated by ridge.py except for getting the bigger histogram

# Max/min of observer slice
def obs_range(
        dataset: np.ndarray
    ) -> np.ndarray:
    '''
    Calculate mins and maxs of dataset.
    '''
    mins = np.amin(dataset, axis=2)
    maxs = np.amax(dataset, axis=2)
    return np.dstack((mins, maxs))

def opa_hist(
        dataset: np.ndarray,
        opa_slices: int
    ) -> np.ndarray:
    '''
    Calculate histogram of dataset for observers and slices.
    '''
    return lib.bucket(dataset, opa_slices)

# opa_hist_ridge(stats_low, stats_high, .9)

def opa_hist_ridge(
        treatment: np.ndarray, 
        control: np.ndarray, 
        certainty: int
    ) -> np.ndarray:
    '''
    Find "ridge" values for the treatment and control histograms (i.e. where `treatment >= control * certainty`).
    '''
    is_valid = treatment >= control * certainty
    # If the treatment has nothing, that may not be a fail, but it certainly isn't success
    is_valid[treatment == 0] = False
    return np.apply_along_axis(np.argmax, 1, is_valid)

def run_dataset(
        dataset: np.ndarray, 
        axs: plt.Axes,
        opa_slices: int = 3, 
        color = "gray"
    ) -> None:
    '''
    Run dataset for either histogram or normal ditribution graph.
    '''
    ranges = obs_range(dataset)
    hist = np.transpose(opa_hist(dataset, opa_slices))[::-1]

    num_obs = hist.shape[2]
    num_cases = hist.shape[1]
    xs = np.arange(num_cases)
    bins = np.arange(num_cases)
    for obs in range(num_obs):
        # Graph ranges
        axis_col = obs
        ax = axs[0][axis_col]
        ax.plot(xs, ranges[obs, :, 0], color=color)
        ax.plot(xs, ranges[obs, :, 1], color=color)

        for opa in range(opa_slices):
            axis_slice = opa + 1
            ax = axs[axis_slice][axis_col]
            ax.hist(
                bins, 
                bins=bins, 
                weights=hist[opa, :, obs], 
                align="left", 
                color=color, 
                alpha=.5
            )
    return np.amax(hist)

def run_ridge(
        datasets: typ.Union[typ.Tuple[np.ndarray, np.ndarray], list[np.ndarray]], 
        observer_slices: int = 3, 
        opa_slices: int = 3,
        certainty: int = 9
    ) -> None:
    '''
    datasets: treatment, control
    '''

    # ranges = [obs_range(dataset, observer_slices) for dataset in datasets]
    treatment = opa_hist(datasets[0], observer_slices, opa_slices, buckets=opa_slices)
    control = opa_hist(datasets[1], observer_slices, opa_slices, buckets=opa_slices)
    ridge = np.transpose(opa_hist_ridge(treatment, control, certainty))[::-1]

    np.savetxt("ridge.csv", ridge, fmt="%d", delimiter=",")

exp = "prostate_reader/"
group = "_5class"
dataset_names = ["./data/" + exp + "assisted" + group + ".npy",
                 "./data/" + exp + "unassisted" + group + ".npy"
                 ]
datasets = [np.transpose(lib.data_reader(set)) for set in dataset_names]

num_obs = datasets[0].shape[0]
num_cases = datasets[0].shape[1]
opa_slices = 10

colors = ["red", "green"]
fig, axs = plt.subplots(nrows=(opa_slices + 1), ncols=num_obs, 
                        figsize=(num_obs * 1.5, (opa_slices + 1) * 1.5), 
                        squeeze=False
                        )

max = -1
for i in range(len(datasets)):
    test = run_dataset(datasets[i], axs, opa_slices, color=colors[i])
    max = max if max > test else test

## Adjust plot parameters
# Hide the axes
for row in axs:
    for ax in row:
        # Hide labels
        ax.tick_params(
            axis="both",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
            labeltop=False,
            labelbottom=False,
            labelleft=False,
            labelright=False,
        )
        # Hide spines
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        # Adjust limits
        ax.set_xlim(0, num_cases)

# Specifics for the OPAs
for row in axs[1:]:
    for ax in row:
        ax.set_ylim(0, max)

# Show edge axes
ax = axs[0][0]
ax.spines[["left"]].set_visible(True)
ax.yaxis.set_tick_params(
    left=True,
    labelleft=True)

# Left column of OPAs
for opa in range(opa_slices):
    ax = axs[opa_slices - opa][0]
    ax.spines[["left"]].set_visible(True)
    ax.yaxis.set_tick_params(
        left=True,
        labelleft=True)
    ax.set_ylabel(str(round(opa/opa_slices, 2)) + " - " + str(round((opa+1)/opa_slices, 2)))

# Bottom row of OPAs
for obs in range(num_obs):
    ax = axs[opa_slices][obs]
    ax.spines[["bottom"]].set_visible(True)
    ax.xaxis.set_tick_params(
        bottom=True,
        labelbottom=True
    )
    ax.set_xlabel(obs + 2)
plt.tight_layout()

fig.text(0, .5, "Overall Proportion Agreement", 
         rotation="vertical",
         size="xx-large",
         horizontalalignment="center",
         verticalalignment="center",
        )
fig.text(.5, 0, "Number of Observers", 
         rotation="horizontal",
         size="xx-large",
         horizontalalignment="center",
         verticalalignment="center",
         )

plt.savefig("./results/" + exp + "hist" + group + ".png",
            bbox_inches="tight", transparent=False, dpi=100)
