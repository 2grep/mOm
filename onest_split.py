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
import lib


file_names = []

dataset_names = ["assisted.npy", "unassisted.npy"]
# (datasets, cases, observers - 1, unique surfaces)
datasets = [np.transpose(lib.data_reader(set)) for set in dataset_names]

# Splitting along observers and OPAs 
# (e.g. 6, 12, & 18 observers x .33, .66, & .99 OPAs -> 
#       6 observers for min/max + .33 OPA stats + .66 OPA stats + .99 OPA stats ...)

observer_slices = 3
opa_slices = 3

colors = ["red", "green"]
color = colors[0]
fig, axs = plt.subplots(nrows=observer_slices, ncols=(opa_slices + 1), squeeze=False)

dataset = datasets[0]


# Max/min of observer slice
def obs_range(dataset, observer_slices):
    # max number of observers / number of slices
    obs_step = dataset.shape[0] // observer_slices
    mins = np.amin(dataset[obs_step::obs_step], axis=2)
    maxs = np.amax(dataset[obs_step::obs_step], axis=2)
    return np.dstack((mins, maxs))

def opa_stats(dataset, observer_slices, opa_slices, buckets=100):
    # max number of observers / number of slices
    obs_step = dataset.shape[0] // observer_slices
    # max number of buckets / number of slices
    bucket_step = buckets // opa_slices

    bucketed = lib.bucket(dataset, buckets)[obs_step::obs_step, :, bucket_step::bucket_step]
    means = np.average(bucketed, axis=1)
    std_devs = np.std(bucketed, axis=1)
    return np.dstack((means, std_devs))

def run_dataset(dataset, axs, observer_slices=3, opa_slices=3):
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

run_dataset(dataset, axs, observer_slices, opa_slices)

plt.show()



















# observer_steps = 3
# total_number_of_observer_pairs = 19
# observer_step_size = total_number_of_observer_pairs // observer_steps
# case_steps = 3
# total_number_of_cases = 240
# case_step_size = total_number_of_cases // case_steps

# # 3D list of DataFrames
# # (row, column, dataset)
# analyses = []
# datasets_length = len(datasets)
# for observer_num in range(observer_step_size, total_number_of_observer_pairs, observer_step_size):
#     observer_row = []
#     case_opa_cross_section_item = []
#     for dataset_index in range(datasets_length):
#         case_opa_cross_section_item.append(pd.DataFrame({
#             "max": np.amax(datasets[dataset_index][:, :, observer_num], axis=0),
#             "min": np.amin(datasets[dataset_index][:, :, observer_num], axis=0),
#         }))
#     observer_row.append(case_opa_cross_section_item)

#     for case_num in range(case_step_size, total_number_of_cases, case_step_size):
#         case_stats_item = []
#         for dataset_index in range(datasets_length):
#             print(datasets[dataset_index][:, :, observer_num].shape)
#             case_stats_item.append(pd.DataFrame({
#                 "mean": np.average(datasets[dataset_index][:, case_num, observer_num], axis=0),
#                 "std": np.std(datasets[dataset_index][:, case_num, observer_num], axis=0)
#             }))
#         observer_row.append(case_stats_item)
        
#     analyses.append(observer_row)

# colors = ["red", "green"]

# fig, axs = plt.subplots(nrows=observer_steps, ncols=(case_steps + 1), squeeze=False)

# # Graph cases x OPA for each slice

# xrange = np.arange(0, 1, .001)
# for row in range(len(analyses)):
#     for dataset in range(datasets_length):
#         color = colors[dataset % len(colors)]
#         analyses[row][0][dataset]["max"].plot.line(
#             style="-",
#             color=color,
#             fillstyle="none",
#             linewidth=1,
#             ax=axs[row][0]
#         )
#         analyses[row][0][dataset]["min"].plot.line(
#             style="-",
#             color=color,
#             fillstyle="none",
#             linewidth=1,
#             ax=axs[row][0]
#         )
    
#         for graph in range(case_steps):
#             print(analyses[row][1+graph][dataset])

#             axs[row][graph].plot(
#                 xrange,
#                 norm.pdf(xrange, analyses[row][1+graph][dataset].loc[case_step_size * (graph + 1), "mean"],
#                          analyses[row][1+graph][dataset].loc[case_step_size * (graph + 1) - 1, "std"]),
#                 color=color)

# plt.show()