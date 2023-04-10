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

# Max/min of observer slice
def obs_range(dataset, observer_slices):
    '''
    Calculate mins and maxs of dataset.
    '''
    # max number of observers / number of slices
    obs_step = dataset.shape[0] // observer_slices
    mins = np.amin(dataset[obs_step::obs_step], axis=2)
    maxs = np.amax(dataset[obs_step::obs_step], axis=2)
    return np.dstack((mins, maxs))

def opa_hist(dataset, observer_slices, opa_slices, buckets=100):
    '''
    Calculate histogram of dataset for observers and slices.
    '''
    # max number of observers / number of slices
    obs_step = dataset.shape[0] // observer_slices
    # max number of buckets / number of slices
    bucket_step = buckets // opa_slices

    return lib.bucket(dataset, buckets)[obs_step::obs_step, :, bucket_step::bucket_step]

def opa_stats(dataset, observer_slices, opa_slices, buckets=100):
    '''
    Calculate statistics (means and standard deviations) of dataset.
    '''
    bucketed = opa_hist(dataset, observer_slices, opa_slices, buckets)
    means = np.average(bucketed, axis=1)
    std_devs = np.std(bucketed, axis=1)
    return np.dstack((means, std_devs))

def run_dataset(dataset, axs, observer_slices=3, opa_slices=3, color="gray", method="hist"):
    if method == "hist":
        ranges = obs_range(dataset, observer_slices)
        # * buckets >= opa_slices
        hist = opa_hist(dataset, observer_slices, opa_slices, buckets=10)

        xs = np.arange(0, dataset.shape[1])
        bins = np.arange(0, dataset.shape[1], 5)
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
                axs[axis_slice][axis_row].set_ylim(0, 5000)
                axs[axis_slice][axis_row].set_frame_on(False)
        
        # Adjust labels for edges
        for ax in axs[:, 0]:
            plt.setp(ax.get_yticklabels(), visible=True)

        for ax in axs[len(axs) - 1, :]:
            plt.setp(ax.get_xticklabels(), visible=True)

    if method == "norm":
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


dataset_names = ["assisted.npy", "unassisted.npy"]
datasets = [np.transpose(lib.data_reader(set)) for set in dataset_names]

observer_slices = 18
opa_slices = 10

colors = ["red", "green"]
fig, axs = plt.subplots(ncols=observer_slices, nrows=(opa_slices + 1), squeeze=False, sharex=True)

for i in range(len(datasets)):
    run_dataset(datasets[i], axs, observer_slices, opa_slices, color=colors[i])

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