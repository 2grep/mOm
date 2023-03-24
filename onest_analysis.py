#!usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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
parser.add_argument("case_observers", metavar="data", help="Path for data to run analysis on", nargs="+", type=pd.read_csv)
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

args = parser.parse_args()

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
    observer_lists = []
    all_observers = list(case_observer_matrix.columns)

    for new_curve in range(unique_curves):
        print("Running curve: ", new_curve)
        random.shuffle(all_observers)
        observers_for_this_curve = all_observers[:O_max]
        # Reshuffle observers until we get something new
        while observers_for_this_curve in observer_lists:
            random.shuffle(all_observers)
            observers_for_this_curve = all_observers[:O_max]

        observer_lists.append(observers_for_this_curve.copy())
       
        curve = []

        for index in range(2, len(observers_for_this_curve)):
            curve.append(overall_proportion_agreement(case_observer_matrix, observers_for_this_curve[:index], fractional))

        onest = pd.concat([onest, pd.Series(curve, index=range(2, len(curve) + 2))], ignore_index=False, axis=1)

    return onest

if args.model == "onest":
    # Convert case_observer matrices to OPAs
    data_df = []
    for case_observer_matrix in args.case_observers:
        # ? How should we structure the call for unique_curves and O_max to be obvious and versatile
        # Ideas:
        # data_1 [unique_curves_1] [o_max_1] data_2 [unique_curves_2] [o_max_2] ...
        # data_1 data_2 ... [--unique_curves uc_1 uc_2 ...] [--o_max om_1 om_2 ...]
        # --data_set data_1 [unique_curves_1] [o_max_1] --data_set data_1 [unique_curves_1] [o_max_1] ...

        unique_curves = 100
        o_max = len(case_observer_matrix.columns)
        df = onest(case_observer_matrix, unique_curves, o_max, args.fractional)
        if args.describe:
            # Desribe as mean, min, max if desired
            df = df.apply(pd.DataFrame.describe, axis=1)[["mean", "min", "max"]]

        data_df.append(df)

    
    opas = data_df[0].plot.line(
        style="-" if args.describe else "o-", 
        color=args.colors[0], 
        legend=False, 
        fillstyle="none", 
        linewidth=1 if args.describe else .5
    )
    counter = 0
    for data in data_df[1:]:
        counter += 1
        data.plot.line(
            style="-" if args.describe else "o-", 
            color=args.colors[counter % len(data_df)], 
            legend=False, 
            fillstyle="none", 
            linewidth=1 if args.describe else .5,
            ax=opas
        )

    opas.xaxis.set_major_locator(MultipleLocator(6))
    opas.xaxis.set_major_formatter('{x:.0f}')
    opas.set_xlim([data_df[0].index[0], data_df[0].index[-1]])
    opas.set_ylim([0, 1])

    plt.show()
