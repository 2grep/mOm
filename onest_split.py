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
        dataset: np.ndarray, 
        observer_slices: int
    ) -> np.ndarray:
    '''
    Calculate mins and maxs of dataset.
    '''
    # max number of observers / number of slices
    obs_step = dataset.shape[0] // observer_slices
    mins = np.amin(dataset[obs_step::obs_step], axis=2)
    maxs = np.amax(dataset[obs_step::obs_step], axis=2)
    return np.dstack((mins, maxs))

def opa_hist(
        dataset: np.ndarray, 
        observer_slices: int, 
        opa_slices: int, 
        buckets: int = 100
    ) -> np.ndarray:
    '''
    Calculate histogram of dataset for observers and slices.
    '''
    # max number of observers / number of slices
    obs_step = dataset.shape[0] // observer_slices
    # max number of buckets / number of slices
    bucket_step = buckets // opa_slices
    print(buckets, bucket_step)
    return lib.bucket(dataset, buckets)[::obs_step, :, ::bucket_step]

def opa_stats(
        dataset: np.ndarray, 
        observer_slices: int, 
        opa_slices: int, 
        buckets: int = 100
    ) -> np.ndarray:
    '''
    Calculate statistics (means and standard deviations) of dataset.
    '''
    bucketed = opa_hist(dataset, observer_slices, opa_slices, buckets)
    means = np.average(bucketed, axis=1)
    std_devs = np.std(bucketed, axis=1)
    return np.dstack((means, std_devs))

# opa_hist_ridge(stats_low, stats_high, .9)

def opa_hist_ridge(
        treatment: np.ndarray, 
        control: np.ndarray, 
        certainty: int
    ) -> np.ndarray:
    '''
    Find "ridge" values for the treatment and control histograms (i.e. where treatment >= control * certainty).
    '''
    is_valid = treatment >= control * certainty
    # If the treatment has nothing, that may not be a fail, but it certainly isn't success
    is_valid[treatment == 0] = False
    return np.apply_along_axis(np.argmax, 1, is_valid)

def run_dataset(
        dataset: np.ndarray, 
        axs: plt.Axes, 
        observer_slices: int = 3, 
        opa_slices: int = 3, 
        color: mpl.Color = "gray", 
        method: str = "hist"
    ) -> None:
    '''
    Run dataset for either histogram or normal ditribution graph.
    '''
    if method == "hist":
        ranges = obs_range(dataset, observer_slices)
        # * buckets >= opa_slices
        hist = opa_hist(dataset, observer_slices, opa_slices, buckets=opa_slices)

        xs = np.arange(0, dataset.shape[1])
        bins = np.arange(0, dataset.shape[1], 1)
        for row in range(observer_slices):
            # Graph ranges
            # axis_row = observer_slices - row - 1
            axis_row = row
            axs[0][axis_row].plot(xs, ranges[row][:, 0], color=color)
            axs[0][axis_row].plot(xs, ranges[row][:, 1], color=color)

            for opa_slice in range(opa_slices):
                axis_slice = opa_slices - opa_slice
                axs[axis_slice][axis_row].hist(
                    xs, bins=bins, weights=hist[row, : , opa_slice], 
                    align="left", color=color, alpha=.5)
                axs[axis_slice][axis_row].set_ylim(0, 1000)

    elif method == "norm":
        ranges = obs_range(dataset, observer_slices)
        stats = opa_stats(dataset, observer_slices, opa_slices)

        xs = np.arange(0, dataset.shape[1])
        for row in range(observer_slices):
            # Graph ranges
            axs[row][0].plot(xs, ranges[row][:, 0], color=color)
            axs[row][0].plot(xs, ranges[row][:, 1], color=color)

            # Graph stats
            for opa_slice in range(opa_slices):
                axs[row][opa_slice + 1].plot(
                    xs, 
                    norm.pdf(
                        xs, 
                        stats[row][opa_slice][0], 
                        stats[row][opa_slice][1]
                    ), 
                    color=color)

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

dataset_names = ["assisted_3case.npy", "unassisted_3case.npy"]
datasets = [np.transpose(lib.data_reader(set)) for set in dataset_names]


observer_slices = 18
opa_slices = 10
# hist_high = opa_hist(datasets[0], observer_slices, opa_slices, opa_slices)
# hist_low = opa_hist(datasets[1], observer_slices, opa_slices, opa_slices) 

colors = ["red", "green"]
fig, axs = plt.subplots(ncols=observer_slices, nrows=(opa_slices + 1), squeeze=False, sharex=True, figsize=(observer_slices * 1.5, (opa_slices + 1) * 1.5))

# fig, ax = plt.subplots()

run_ridge(datasets, observer_slices, opa_slices, 9)


for i in range(len(datasets)):
    run_dataset(datasets[i], axs, observer_slices, opa_slices, color=colors[i])

plt.savefig("mom.png", bbox_inches="tight")
plt.show()