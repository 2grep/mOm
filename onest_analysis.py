#!usr/bin/env python3
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LinearLocator)
import random as random
from pprint import pprint

## ARGUMENTS ##
# TODO: convert unique_curves and o_max to inputted values
# TODO: add verbosity option
# 0 - print nothing, just make the graph
# 1 - print curve generation progress (preferably as loading bar)
#       Progress: |█████████████████████████████████████████████-----| 90/100 Curve(s) Complete 
#       (https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/13685020)
# TODO: defeat entropy and make dev mode where it uses a pre-made set of obervers with each run rather than regenerating each time
# TODO!: add in ga and esi support

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("dataset_names", metavar="data", help="Path for data to run analysis on", nargs="+")
parser.add_argument("-m", "--model", help="""Model for analysis:
    onest: Observers Needed to Evaluate a Subjective Test,
    ga: Generalized Accuracy -- future,
    esi: Error Severity Index -- future""", 
    dest="model", 
    choices=[
        "onest",
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
    fname, fext = os.path.splitext(file_name)
    file_names.append(fname)
         
    if fext == ".pkl":
        datasets_from_cache = True
        return pd.read_pickle(file_name)
    else:
        return pd.read_csv(file_name)

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
        first = case[observers[0]]
        for observer in observers[1:len(observers)]:
            # if the observations are different
            if case[observer] != first:
                return 0

        return 1
    
    else:
        return case[observers].value_counts().max() / len(observers)
    
def random_unique_permutations(array, max_choices=-1):
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
    case_agreements = case_observer_matrix.apply(match, args=args, axis=1)
    # number of full row-matches / number of cases
    return case_agreements.sum() / len(case_observer_matrix.index)


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
    ax.set_xlim([dataset_onest_analyses[0].index[0], dataset_onest_analyses[0].index[-1]])
    ax.set_xlabel("Number of Observers")
    ax.set_ylim([0, 1])
    ax.set_ylabel("Overall Proportion Agreement")
    ax.legend(ax.get_lines()[::3], args.labels if args.labels != None else file_names)

    # plt.show()

elif args.model == "onest_cummulative":
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.

    observers_min = args.datasets[0].index[0]
    observers_max = args.datasets[0].index[-1]
    observers_axis = np.arange(observers_min, observers_max)
    cases_min = args.datasets[0].columns[0]
    cases_max = args.datasets[0].columns[-1]
    cases_axis = np.arange(cases_min, cases_max)
    observers_axis, cases_axis = np.meshgrid(observers_axis, cases_axis)

    # Run ONEST with O observers and C cases (for each cell in [observers_axis x cases_axis])
    

    proportion_agreement_axis = observers_axis

    # Plot the surface.
    surf = ax.plot_surface(observers_axis, cases_axis, proportion_agreement_axis, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
 
    plt.show()