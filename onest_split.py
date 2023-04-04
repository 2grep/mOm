import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    MultipleLocator, AutoMinorLocator, LinearLocator)
import numpy as np
import os
import pandas as pd
from scipy.stats import norm
from pprint import pprint
import random as random
import time


file_names = []
def data_reader(file_name):
    global datasets_from_cache
    fname, fext = os.path.splitext(file_name)
    file_names.append(fname)

    print("Name:", fname, "Ext:", fext)
    if fext == ".pkl":
        datasets_from_cache = True
        return pd.read_pickle(file_name)
    elif fext == ".npy":
        datasets_from_cache = True
        return np.load(file_name)
    else:
        data = pd.read_csv(file_name)
        print(data)
        return data

dataset_names = ["assisted.npy", "unassisted.npy"]
datasets = [data_reader(set) for set in dataset_names]

dataset_surfaces = []
observer_steps = 3
total_number_of_observer_pairs = 19
observer_step_size = total_number_of_observer_pairs // observer_steps
for dataset in datasets:
    for observer_num in range(observer_step_size, total_number_of_observer_pairs, observer_step_size):
        dataset_surfaces.append(pd.DataFrame({
            "max": np.amax(dataset[:, :, observer_num], axis=0),
            "min": np.amin(dataset[:, :, observer_num], axis=0),
            "mean": np.average(dataset[:, :, observer_num], axis=0),
            "std": np.std(dataset[:, :, observer_num], axis=0),
        }))

# assisted_df = pd.DataFrame(dataset_surfaces[0,:,:, observer_step - 1]).T
# unassisted_df = pd.DataFrame(dataset_surfaces[1,:,:, observer_step - 1]).T

colors = ["red", "green"]

length = len(dataset_surfaces)
observer_steps = length // len(datasets)
case_steps = 3
total_number_of_cases = 240
case_step_size = total_number_of_cases // case_steps
fig, axs = plt.subplots(nrows=observer_steps, ncols=(case_steps + 1), squeeze=False)
for case_plot in range(observer_steps):
    for dataset in range(0, length, observer_steps):
        color = colors[dataset % len(colors)]
        index = dataset + case_plot
        dataset_surfaces[index]["max"].plot.line(
            style="-",
            color=color,
            fillstyle="none",
            linewidth=1,
            ax=axs[case_plot][0]
        )
        dataset_surfaces[index]["min"].plot.line(
            style="-",
            color=color,
            fillstyle="none",
            linewidth=1,
            ax=axs[case_plot][0]
        )
        axs[case_plot][0].set_ylim([0, 1])
        x = np.arange(0, 1, .001)
        axs[case_plot][1].plot(
            x, 
            norm.pdf(x, dataset_surfaces[index].loc[case_step_size, "mean"], dataset_surfaces[index].loc[case_step_size - 1, "std"]),
            color=color)

        axs[case_plot][2].plot(
            x, norm.pdf(x, dataset_surfaces[index].loc[case_step_size * 2, "mean"], dataset_surfaces[index].loc[case_step_size * 2 - 1, "std"]),
            color=color)

        axs[case_plot][3].plot(
            x, 
            norm.pdf(x, dataset_surfaces[index].loc[case_step_size * 3 - 1, "mean"], dataset_surfaces[index].loc[case_step_size * 3 - 1, "std"]),
            color=color)



plt.show()