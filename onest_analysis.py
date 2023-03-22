import numpy as np
import pandas as pd
import matplotlib as plt

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

case_observer_matrix = pd.read_csv('./prostate_assisted.csv')
print(case_observer_matrix.nunique())
case_observer_matrix.head(15)
confusion_matrix = pd.crosstab(case_observer_matrix['GU_majority_Ground_truth'], case_observer_matrix['B'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

## FUNCTIONS ##
# Written in the style of David Jin wrote these, originally here: 
# https://colab.research.google.com/drive/10By9_PZLvDY9EAa-n_tt8RGvSfoaQO8x

observers = list(case_observer_matrix.columns)[1:]
observers = observers[:(len(observers) - 2)]
print(observers)
def match(series, observers):
    '''
    All observer columns of series matching
    1 if all observers match, else 0
    '''
    prev = observers[0]
    for observer in observers[1:(len(observers) - 1)]:
        # if the observations are different
        if series[observer] != series[prev]:
            return 0
        prev = observer
    return 1

    # Alternative possible solution, but harder to extend to fractional agreements:
    a = series[observers].to_numpy()
    return 1 if (a[0] == a).all() else 0


def overall_proportion_agreement(case_observer_matrix, observers):
    '''
    Overall proportion agreement (OPA) takes in a N x O_m matrix of N cases rated by O_m observers and returns a measure of the overall agreement for O_x observers.
    '''
    #                                                             number of full row-matches / number of cases
    return case_observer_matrix.apply(match, args=(observers,), axis=1).sum() / len(case_observer_matrix.index)

print(overall_proportion_agreement(case_observer_matrix, observers))

# stop here, we will try more tomorrow.


def onest(case_observer_matrix, C, O_max):
    '''
    onest takes in the (case X observer m) matrix and the desired number of iterations C, and returns a C x O_m-1 matrix of OPAs
    [
     [opa1_1, opa1_2, ... opa1_Om-2, opa1_Om-1],
     [opa2_1, opa2_2, ... opa2_Om-2, opa2_Om-1]
      ....
     [opaC_1, opaC_2, ... opaC_Om-2, opaC_Om-1]
    ]
    '''

    # TODO:
    # create C random, unique orders of O_max observers, e.g. ['C', 'Q', 'L', 'R', etc]
    # cumulatively run OPA for each observer in each random order e.g. ['C', 'Q'] -> ['C', 'Q', 'L'] -> ['C', 'Q', 'L', 'R'] -> ...
    # There are probably some safety conditions:
    #    C shouldn't exceed factorial of O_m
    #    O_max can't exceed O_m
    # Possible min-max epsilon or min/max/median plateau epsilon to allow quitting before hitting O_max

## MAIN: Print graphs##
