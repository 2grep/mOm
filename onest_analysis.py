#!usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, LinearLocator)
import numpy as np
import pandas as pd
import time
import lib

# TODO: convert from pandas DataFrames to NumPy nd_arrays for ALL calculations

## ARGUMENTS ##
# TODO: convert unique_curves and o_max to inputted values
# TODO: add output file for resulting images and caches
# TODO: add verbosity option
# 0 - print nothing, just make the graph
# 1 - print curve generation progress (preferably as loading bar)
#       Progress: |█████████████████████████████████████████████-----| 90/100 Curve(s) Complete
#       (https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/13685020)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("dataset_names", metavar="data", help="Path for data to run analysis on", nargs="+")
parser.add_argument("-m", "--model", help="""Model for analysis:
    onest: Observers Needed to Evaluate a Subjective Test,
    ga: Generalized Accuracy -- future,
    esi: Error Severity Index -- future""", 
    dest="model", 
    choices=[
        "onest",
        "sarape"
    ], required=True)
parser.add_argument("-d", "--statistical_analysis", help="Only graph lines for max, min, and mean of each number of observers", dest="describe", action="store_true")
# TODO: restrict color choices to `matplotlib` colors and colormaps (colormaps only for 3d models)
parser.add_argument("-c", "--color", help="matplotlib colors for each set of data; loops number of colors is less than number of data files", dest="colors", nargs="+", default=["tab:gray"])
parser.add_argument("-l", "--labels", help="Assign labels for each dataset to use in legend", dest="labels")
parser.add_argument("--cache", help="If flagged, caches data after processing", dest="cache", action="store_true")

args = parser.parse_args()

file_names = []
file_exts = []

## FUNCTIONS ##
# Written in the style of David Jin wrote these, originally here:
# https://colab.research.google.com/drive/10By9_PZLvDY9EAa-n_tt8RGvSfoaQO8x

def overall_proportion_agreement(case_observer_matrix: np.ndarray) -> float:
    '''
    Overall proportion agreement (OPA) takes in a N x O_m matrix of N cases rated by O_m observers and 
    returns a measure of the overall agreement for observers.

    Assumes observers are in first dimension
    '''
    # Check for matches across rows
    case_agreements = np.apply_along_axis(lib.match, 0, case_observer_matrix)
    # number of full row-matches / number of cases
    return case_agreements.sum() / case_observer_matrix.shape[1]

def sarape(
        case_observer_matrix: np.ndarray, 
        unique_surfaces: int
    ) -> np.ndarray:
    # TODO: What does "unique" surfaces mean? Define uniqueness.
    '''
    Calculate SARAPE model on `case_observer_matrix` to generate `unique_surfaces` number of samples from
    the full space of samples. 

    Parameters
    ----------
    case_observer_matrix : observers O x cases c matrix
        ```
        [[A0, A1, ..., Ac],
         [B0, B1, ..., Bc],
         ...
         [O0, O1, ..., Oc]]
        ```
    unique_surfaces : number of surfaces to run
        More surfaces is a better sample of the full space but takes longer. "Unique" meaning the orders 
        of rows and columns of case_observer_matrix are independently (i.e. uniqueness in row is unrelated
        to that of columns) unique orderings.
    
    Returns
    -------
    sarape : surfaces calculated with a shape of `(unique surfaces, cases, total observers - 1)`
    '''
    # Generators for observers and cases

    num_obs = case_observer_matrix.shape[0]
    num_cases = case_observer_matrix.shape[1]
    obs_gen = lib.random_unique_permutations(np.arange(num_obs))
    cases_gen = lib.random_unique_permutations(np.arange(num_cases))

    space = []
    for new_surface in range(unique_surfaces):
        if new_surface % 10 == 0:
            print("Running surface:", new_surface)

        surf_obs = next(obs_gen)
        surf_cases = next(cases_gen)

        # cases x observers
        grid = []
        for case_ind in range(num_cases):
            reduced_cases = surf_cases[:case_ind + 1]

            observer_opas = []
            for obs_ind in range(2, num_obs + 1):
                reduced_obs = surf_obs[:obs_ind]

                # We need to index twice bc of broadcasting isues between the two indices
                matrix = case_observer_matrix[reduced_obs][:, reduced_cases]
                opa = overall_proportion_agreement(matrix)

                observer_opas.append(opa)
            grid.append(observer_opas)
        space.append(grid)
    return np.array(space, copy=False)

def onest(
        case_observer_matrix: np.ndarray, 
        unique_curves: int
    ) -> np.ndarray:
    '''
    Parameters
    ----------
    case_observer_matrix : observers O x cases c matrix
        ```
        [[A0, A1, ..., Ac],
         [B0, B1, ..., Bc],
         ...
         [O0, O1, ..., Oc]]
        ```
    unique_curves : number of curves to runs

    Returns
    -------
    onest : 
    '''
    onest = []
    num_obs = case_observer_matrix.shape[0]
    observers_generator = lib.random_unique_permutations(np.arange(num_obs))

    for new_curve in range(unique_curves):
        if new_curve % 10 == 0:
            print("Running curve: ", new_curve)

        curve_obs = next(observers_generator)

        ## Generate single onest curve
        curve = []
        for ind in range(2, num_obs + 1):
            # num of observers x OPA point on the ONEST curve
            reduced_obs = curve_obs[:ind]
            matrix = case_observer_matrix[reduced_obs]
            opa = overall_proportion_agreement(matrix)
            curve.append(opa)

        onest.append(curve)
    return np.array(onest, copy=False)


args.datasets = np.asarray([lib.data_reader(set, names=file_names, exts=file_exts)[:, :20] for set in args.dataset_names])

datasets_from_cache = [".pkl", ".npy"] in file_exts

if args.model == "onest":
    unique_curves = 10000
    ## Convert case_observer matrices to OPAs (i.e. One set (each item in dataset_onest_analyses) of curves for each dataset)
    if not datasets_from_cache:
        counter = 0
        onest_analyses: list[np.ndarray] = []
        for cases_x_observers_matrix in args.datasets:
            analysis = onest(np.transpose(cases_x_observers_matrix), unique_curves)

            onest_analyses.append(analysis)

            if args.cache:
                # TODO: Need to encode certain information in args such as if -d is flagged
                np.save(file_names[counter] + ".npy", analysis)
                counter += 1
        onest_analyses = np.asarray(onest_analyses)
        onest_analyses = np.transpose(onest_analyses, (0, 2, 1))
    
    else:
        onest_analyses = args.datasets

    # ! put this in numpy terms
    if args.describe:
        # Desribe as min, mean, max if desired
        onest_curves = np.dstack((
            np.apply_along_axis(np.amin,    2, onest_analyses), 
            np.apply_along_axis(np.average, 2, onest_analyses), 
            np.apply_along_axis(np.amax,    2, onest_analyses)
        ))

    else:
        onest_curves = onest_analyses

    ## Plot each analysis
    fig, ax = plt.subplots()
    obs_range = (2, onest_analyses.shape[1] + 1)
    xs = np.arange(obs_range[0], obs_range[1] + 1)
    for dataset in range(onest_curves.shape[0]):
        ax.plot(
            xs, onest_curves[dataset],
            ("-" if args.describe else "o-"),
            color=args.colors[dataset % len(args.colors)],
            fillstyle="none",
            linewidth=1 if args.describe else .5
        )

    # adjust plot parameters
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.set_xlim(*obs_range)
    ax.set_xlabel("Number of Observers")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Overall Proportion Agreement")
    ax.legend(ax.get_lines()[::(3 if args.describe else unique_curves)],
              args.labels if args.labels != None else file_names)

    # plt.show()

elif args.model == "sarape":
    # Run ONEST with S observers and C cases (for each cell in [observers_axis x cases_axis])
    if not datasets_from_cache:
        counter = 0
        case_onest_analyses: list[np.ndarray] = []
        unique_surfaces = 1000

        for cases_x_observers_matrix in args.datasets:
            # start = time.time()
            single_analysis = sarape(
                np.transpose(cases_x_observers_matrix), 
                unique_surfaces
            )
            # end = time.time()
            # print("ONEST calculation time:", end - start)

            if args.cache:
                np.save(file_names[counter], np.asarray(single_analysis))
                counter += 1

            case_onest_analyses.append(single_analysis)
        case_onest_analyses = np.asarray(case_onest_analyses)

    else:
        case_onest_analyses = args.datasets

    if args.describe:
        # Get min and max
        dataset_surfaces = []
        for dataset in case_onest_analyses:
            dataset_surfaces.append([
                np.amax(dataset, axis=0), 
                np.amin(dataset, axis=0)]
            )
        dataset_surfaces = np.asarray(dataset_surfaces)
    
    else:
        dataset_surfaces = case_onest_analyses

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    colors = ["coolwarm", "PiYG"]
    obs_range = (2, dataset_surfaces.shape[3] + 1)
    observers_axis = np.arange(obs_range[0], obs_range[1] + 1)
    cases_range = (1, dataset_surfaces.shape[2])
    cases_axis = np.arange(cases_range[0], cases_range[1] + 1)
    observers_axis, cases_axis = np.meshgrid(observers_axis, cases_axis)
    # Plot the surface

    for dataset in range(dataset_surfaces.shape[0]):
        for surface in dataset_surfaces[dataset]:
            ax.plot_surface(
                observers_axis, cases_axis,
                surface,
                cmap=colors[dataset % len(colors)],
                linewidth=0, 
                antialiased=False
            )

    # Customize the z axis.
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.set_xlim(*obs_range)
    ax.set_xlabel("Number of Observers")
    ax.set_ylim(*cases_range)
    ax.set_ylabel("Number of Cases")
    ax.set_zlim(0, 1)
    ax.set_zlabel("Overall Proportion Agreement")

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    plt.show()