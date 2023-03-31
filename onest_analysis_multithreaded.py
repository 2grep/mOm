#!usr/bin/env python3
import argparse
import os
import time
import numpy as np
import functools
import itertools
import operator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LinearLocator)
import random as random
from pprint import pprint

import multiprocessing
import concurrent.futures

## ARGUMENTS ##
# TODO: convert unique_curves and o_max to inputted values
# TODO: add verbosity option
# 0 - print nothing, just make the graph
# 1 - print curve generation progress (preferably as loading bar)
#       Progress: |█████████████████████████████████████████████-----| 90/100 Curve(s) Complete
#       (https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/13685020)
# TODO!: add in ga and esi support

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("dataset_names", metavar="data", help="Path for data to run analysis on", nargs="+")
# TODO: should really call each model from seperate files for clarity
parser.add_argument("-m", "--model", help="""Model for analysis:
    onest: Observers Needed to Evaluate a Subjective Test,
    ga: Generalized Accuracy -- future,
    esi: Error Severity Index -- future""", 
    dest="model", 
    choices=[
        "onest",
        "onest_cummulative",
        "ga",
        "esi"
    ], required=True)
parser.add_argument("-f", "--fractional", help="Use fractional agreement in ONEST model", dest="fractional", action="store_true")
parser.add_argument("-d", "--statistical_analysis", help="Only graph lines for max, min, and mean of each number of observers", dest="describe", action="store_true")
parser.add_argument("-c", "--color", help="matplotlib colors for each set of data; loops number of colors is less than number of data files", dest="colors", nargs="+", default=["tab:gray"])
parser.add_argument("-l", "--labels", help="Assign labels for each dataset to use in legend", dest="labels")
parser.add_argument("--cache", help="If flagged, caches data after processing", dest="cache", action="store_true")

args = parser.parse_args()

datasets_from_cache = False
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


args.datasets = [data_reader(set) for set in args.dataset_names]

## FUNCTIONS ##
# Written in the style of David Jin wrote these, originally here:
# https://colab.research.google.com/drive/10By9_PZLvDY9EAa-n_tt8RGvSfoaQO8x


def match(case, observers, fractional=False):
    '''
    Check if all observers of case match.
    If fractional is set to true, computes fractional agreement: max agreements / number of observers

    Returns: 1 if all observers match, else 0

    '''
    if not fractional:
        ## Python/early quit match
        first = case[observers[0]]
        for observer in observers[1:len(observers)]:
            # if the observations are different
            if case[observer] != first:
                return 0
        return 1

        ## NumPy match
        # return int(np.all(case[observers] == case[observers[0]]))

        ## functools match
        # return functools.reduce(operator.eq, case[observers])

        ## itertools match
        # g = itertools.groupby(case[observers])
        # return next(g, True) and not next(g, False)

        ## all(first == x)
        # iterator = iter(case[observers])
        # try:
        #     first = next(iterator)
        # except StopIteration:
        #     return True
        # return all(first == x for x in iterator)

        ## set length
        # return len(set(case[observers])) == 1


    else:
        return case[observers].value_counts().max() / len(observers)
    
def random_unique_permutations(array, max_choices=-2):
    max_choices += 1
    prev_permutations = []
    while True:
        random.shuffle(array)
        new_permutation = array[:max_choices]
        while new_permutation in prev_permutations:
            random.shuffle(array)
            new_permutation = array[:max_choices]

        yield new_permutation

def overall_proportion_agreement(case_observer_matrix, *args):
    '''
    Overall proportion agreement (OPA) takes in a N x O_m matrix of N cases rated by O_m observers and returns a measure of the overall agreement for observers.
    '''
    # Check for matches across rows
    case_agreements = case_observer_matrix.apply(match, args=args, axis=1)
    # number of full row-matches / number of cases
    return case_agreements.sum() / len(case_observer_matrix.index)


def cases_x_onest(case_observer_matrix, num_unique_surfaces, max_num_cases, max_num_observers, fractional=False):
    # Generators for observers and cases
    all_observers = list(case_observer_matrix.columns)
    all_cases = list(case_observer_matrix.index)
    observers_generator = random_unique_permutations(all_observers, max_num_observers)
    cases_generator = random_unique_permutations(all_cases, max_num_cases)

    space = []

    def _generate_surface(surface_num, surface_cases, surface_observers, space):
        # cases x observers
        opa_grid = []

        for cumulative_case_index in range(1, len(surface_cases) + 1):
            print("Running case:", str(surface_num) + "." + str(cumulative_case_index))
            observer_opas = []

            for cumulative_observer_index in range(2, len(surface_observers)):
                reduced_cases = case_observer_matrix.iloc[surface_cases[:cumulative_case_index]]
                reduced_observers = surface_observers[:cumulative_observer_index]

                opa = overall_proportion_agreement(
                    reduced_cases,
                    reduced_observers,
                    fractional)

                observer_opas.append(opa)

            # print(observer_opas)
            opa_grid.append(observer_opas)

        space.append(opa_grid)

    threads = multiprocessing.cpu_count()
    executor = concurrent.futures.ThreadPoolExecutor(threads)
    futures = {}
    for new_surface in np.arange(num_unique_surfaces):
        print("Running surface:", new_surface)
        surface_cases = next(cases_generator)
        surface_observers = next(observers_generator)
        
        futures[executor.submit(
            _generate_surface,
            new_surface,
            surface_cases,
            surface_observers,
            space
        )] = new_surface
    
    executor.shutdown()

    return space


def onest(case_observer_matrix, unique_curves, O_max, fractional=False):
    '''
    onest takes in the (case X observer m) matrix and the desired number of iterations C, and returns a C x O_m-1 matrix of OPAs
    [
     [opa1_1, opa1_2, ... opa1_Om-2, opa1_Om-1],
     [opa2_1, opa2_2, ... opa2_Om-2, opa2_Om-1]
      ....
     [opaC_1, opaC_2, ... opaC_Om-2, opaC_Om-1]
    ]
    
    unique_curves must be less than {O_m choose O_max} - WILL enter infinite loop if this condition is not held
    We're not checking for this because if this is a problem, you shouldn't be using this.
    '''
    # slicing is exclusive, we assume O_max is inclusive (if you want to use 10 observers, you get 10 observers / 9 OPAs)
    O_max += 1

    onest = pd.DataFrame()
    all_observers = list(case_observer_matrix.columns)
    observers_generator = random_unique_permutations(all_observers, O_max)
    # permutations_time_aggregate = 0
    # onest_calculations_time_aggregate = 0

    for new_curve in range(unique_curves):
        print("Running curve: ", new_curve)
        ## Get the unique random permutaion of observers

        # start = time.time()
        observers_for_this_curve = next(observers_generator)
        # end = time.time()
        # permutations_time_aggregate += end - start

        ## Generate single onest curve

        # start = time.time()
        curve = []
        for index in range(2, len(observers_for_this_curve)):
            curve.append(overall_proportion_agreement(case_observer_matrix, observers_for_this_curve[:index], fractional))
        # end = time.time()
        # onest_calculations_time_aggregate += end - start

        onest = pd.concat([onest, pd.Series(curve, index=range(2, len(curve) + 2))], ignore_index=False, axis=1)
    
    # print(f"Time to generate {O_max} random unique permutations of observsers: ", permutations_time_aggregate)
    # print(f"Time to calculate {O_max} ONEST curves: ", onest_calculations_time_aggregate)

    return onest


if args.model == "onest":
    ## Convert case_observer matrices to OPAs (i.e. One set (each item in dataset_onest_analyses) of curves for each dataset)
    dataset_onest_analyses = []
    if not datasets_from_cache:
        counter = 0
        for cases_x_observers_matrix in args.datasets:
            # ? How should we structure the call for unique_curves and O_max to be obvious and versatile
            # Ideas:
            # data_1 [unique_curves_1] [o_max_1] data_2 [unique_curves_2] [o_max_2] ...
            # data_1 data_2 ... [--unique_curves uc_1 uc_2 ...] [--o_max om_1 om_2 ...]
            # --data_set data_1 [unique_curves_1] [o_max_1] --data_set data_1 [unique_curves_1] [o_max_1] ...
            unique_curves = 100
            o_max = len(cases_x_observers_matrix.columns)
            cases_x_observers_onest_analysis = onest(cases_x_observers_matrix, unique_curves, o_max, args.fractional)
            if args.describe:
                # Desribe as mean, min, max if desired
                cases_x_observers_onest_analysis = cases_x_observers_onest_analysis.apply(pd.DataFrame.describe, axis=1)[["mean", "min", "max"]]

            dataset_onest_analyses.append(cases_x_observers_onest_analysis)

            if args.cache:
                # TODO: Need to encode certain information in args such as if -d is flagged
                cases_x_observers_onest_analysis.to_pickle(file_names[counter] + ".pkl")
                counter += 1
    
    else:
        dataset_onest_analyses = args.datasets

    ## Plot each analysis
    fig, ax = plt.subplots()
    plots = [dataset_onest_analyses[0].plot.line(
        style="-" if args.describe else "o-", 
        color=args.colors[0], 
        # legend=False, 
        label="Label 1",
        fillstyle="none",
        linewidth=1 if args.describe else .5,
        ax=ax
    )]
    counter = 0
    for data in dataset_onest_analyses[1:]:
        counter += 1
        plots.append(
            data.plot.line(
                style="-" if args.describe else "o-",
                color=args.colors[counter % len(args.colors)],
                # legend=False,
                fillstyle="none",
                linewidth=1 if args.describe else .5,
                ax=ax
            )
        )

    # adjust plot parameters
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.set_xlim([dataset_onest_analyses[0].index[0],
                dataset_onest_analyses[0].index[-1]])
    ax.set_xlabel("Number of Observers")
    ax.set_ylim([0, 1])
    ax.set_ylabel("Overall Proportion Agreement")
    ax.legend(ax.get_lines()[::3],
              args.labels if args.labels != None else file_names)

    plt.show()

elif args.model == "onest_cummulative":
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # observers_min = args.datasets[0].index[0]
    # observers_max = args.datasets[0].index[-1]
    observers_min = 2
    observers_max = 20
    observers_axis = np.arange(observers_min, observers_max + 1)
    cases_min = 1
    # cases_max = len(args.datasets[0].columns)
    cases_max = 240
    cases_axis = np.arange(cases_min, cases_max + 1)
    observers_axis, cases_axis = np.meshgrid(observers_axis, cases_axis)

    # Run ONEST with O observers and C cases (for each cell in [observers_axis x cases_axis])

    case_onest_analyses = []
    counter = 0
    if not datasets_from_cache:
        for cases_x_observers_matrix in args.datasets:
            # Get case x observer bounds
            unique_surfaces = 1000
            max_num_cases = len(cases_x_observers_matrix.index)
            max_num_observers = len(cases_x_observers_matrix.columns)

            start = time.time()
            single_analysis = cases_x_onest(
                cases_x_observers_matrix, unique_surfaces, max_num_cases, max_num_observers, args.fractional)
            end = time.time()
            print("ONEST calculation time:", end - start)

            case_onest_analyses.append(single_analysis)

            if args.cache:
                np.save(file_names[counter], np.asarray(single_analysis))
                counter += 1

        # case_onest_analyes = ndarry
        # case_onest_analyes = (len(args.datasets), unique_surfaces, max_num_cases, max_num_observers - 1)
        #                    = (datasets, surfaces, cases, observers - 1)
        case_onest_analyses = np.asarray(case_onest_analyses)

    else:
        # case_onest_analyes = list of ndarrys
        # case_onest_analyes = (len(args.datasets), unique_surfaces, max_num_cases, max_num_observers - 1)
        #                    = (datasets, surfaces, cases, observers - 1)
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
        dataset_surfaces = np.asarray(case_onest_analyses)

    # Plot the surface
    surfs = []
    colors = ["coolwarm", "PiYG"]
    print(dataset_surfaces.shape)
    for dataset in np.arange(dataset_surfaces.shape[0]):
        for surface in dataset_surfaces[dataset]:
            surf = ax.plot_surface(observers_axis, cases_axis, surface, cmap=colors[dataset % len(colors)],
                                linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.set_xlim(2, 20)
    ax.set_xlabel("Number of Observers")
    ax.set_ylim([1, 240])
    ax.set_ylabel("Number of Cases")
    ax.set_zlim(0, 1)
    ax.set_zlabel("Overall Proportion Agreement")

    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()