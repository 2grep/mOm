import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import random as random
from pprint import pprint

## BUILD CONFUSION MATRIX ##
# The main return uses the confusion matrix conf_matrix, however, there are
# potentially many confusion matrices that could be set up, for example the
# Steiner 2020 Fig 2D multi-reader dataset affords the opportunity to generate
# many confusion matrices.  So it may be useful to extract various confusion
# matrices from the data, and swap them into the final conf_matrix as we figure
# out what it is we want to draft.
#
# This crosstab method of constructing a confusion matrix from here:
# https://datatofish.com/confusion-matrix-python/. This works as long as all
# cells are populated. For production, may need to check for the input for 
# empty rows and NaN cells and delete them:
# https://stackoverflow.com/a/17092718

assisted_case_observer_matrix = pd.read_csv('./prostate_assisted.csv')
assisted_case_observer_matrix.head(15)
observers = list(assisted_case_observer_matrix.columns)

unassisted_case_observer_matrix = pd.read_csv('./prostate_unassisted.csv')
unassisted_case_observer_matrix.head(15)

## FUNCTIONS ##
# Written in the style of David Jin wrote these, originally here: 
# https://colab.research.google.com/drive/10By9_PZLvDY9EAa-n_tt8RGvSfoaQO8x

# observers = ['A', 'B', 'C', 'D', 'E']
# values =    ['0', '2', '2', '3', '1']
# case = pd.Series(data=values, index=observers)
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

        # Alternative (probably fater) method if we have MANY observers, just less readable
        # a = case[observers].to_numpy()
        # return 1 if (a[0] == a).all() else 0
    
    else:
        return case[observers].value_counts().max() / len(observers)


def overall_proportion_agreement(case_observer_matrix, observers):
    '''
    Overall proportion agreement (OPA) takes in a N x O_m matrix of N cases rated by O_m observers and returns a measure of the overall agreement for observers.
    '''
    case_agreements = case_observer_matrix.apply(match, args=(observers, False), axis=1)
    # number of full row-matches / number of cases
    return case_agreements.sum() / len(case_observer_matrix.index)

# observers = ['A', 'B', 'C', 'D', 'G', 'H']
# print(overall_proportion_agreement(case_observer_matrix, observers))


def onest(case_observer_matrix, unique_curves, O_max):
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
        random.shuffle(all_observers)
        observers_for_this_curve = all_observers[:O_max]
        # Reshuffle observers until we get something new
        while observers_for_this_curve in observer_lists:
            random.shuffle(all_observers)
            observers_for_this_curve = observers[:O_max]

        observer_lists.append(observers_for_this_curve.copy())
       
        curve = []

        for index in range(2, len(observers_for_this_curve)):
            curve.append(overall_proportion_agreement(case_observer_matrix, observers_for_this_curve[:index]))

        onest = pd.concat([onest, pd.Series(curve, index=range(2, len(curve) + 2))], ignore_index=False, axis=1)

    return onest
    
onest_assisted_df = onest(assisted_case_observer_matrix, 100, 20)
onest_unassisted_df = onest(unassisted_case_observer_matrix, 100, 20)

assisted_description = onest_assisted_df.describe()
pprint(assisted_description)

pprint(onest_assisted_df)   
pprint(onest_unassisted_df)  

lines = onest_assisted_df.plot.line(
    style="o-", 
    color="#B10202", 
    legend=False, 
    fillstyle="none", 
    linewidth=.5
)

onest_unassisted_df.plot.line(
    style="o-", 
    color="#008000", 
    legend=False, 
    fillstyle="none", 
    linewidth=.5, 
    ax=lines
)


lines.xaxis.set_major_locator(MultipleLocator(6))
lines.xaxis.set_major_formatter('{x:.0f}')
lines.set_xlim([onest_assisted_df.index[0], onest_assisted_df.index[-1]])
lines.set_ylim([0, 1])
plt.show()
