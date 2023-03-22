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
        prev = observers[0]
        for observer in observers[1:(len(observers) - 1)]:
            # if the observations are different
            if series[observer] != series[prev]:
                return False
            prev = observer
        return True

def overall_proportion_agreement(case_observer_matrix, observers):
    '''
    Overall proportion agreement (OPA) takes in a N x O_m matrix of N cases rated by O_m observers and returns a measure of the overall agreement for O_x observers.
    '''
    cases = len(case_observer_matrix.index)
    
    observers = ["A", "B", "C"]
        
    return case_observer_matrix.apply(match, args=(observers,), axis=1).value_counts()[True] / cases

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


     

def overall_percent_agreement(case_observer_matrix, observers):
    '''
    overall_percent_agreement is a synonym for overall_proportion_agreement()
    ''' 
    return overall_proportion_agreement(case_observer_matrix, observers)

def generalized_precision(conf_matrix, weights, label_num):
    '''
    Generalized precision is computed for a specific label. For a given label, it computes a measure of all the label's predicted instances which were correctly labelled.
    '''
    row = conf_matrix[label_num]

    if sum(row) == 0:
        raise("No elements with that predicted label")

    score = 0
    show_work = "("
    for i, x in enumerate(row):
        score += x * weights[label_num][i]

        # to print out the work
        show_work += ("{} * {:.2f}").format(x, weights[label_num][i])
        if i != len(row) - 1:
            show_work += (" + ")
        else:
            show_work += (") / ({}) = ").format(sum(row))
        # to print out the work

    gen_prec = score / sum(row)
    print(show_work, gen_prec)
    return gen_prec

## BUILD WEIGHTS MATRICES ##
# Similar to conf_matrix, it may be useful to have various weights matrices to
# swap in.

weights_naive = [[1,0,0,0,0], 
                 [0,1,0,0,0], 
                 [0,0,1,0,0], 
                 [0,0,0,1,0], 
                 [0,0,0,0,1]]

weights_sym_shallow = [[1.0,0.9,0.5,0.1,0.0], 
                       [0.9,1.0,0.9,0.5,0.1], 
                       [0.5,0.9,1.0,0.9,0.5], 
                       [0.1,0.5,0.9,1.0,0.9], 
                       [0.0,0.1,0.5,0.9,1.0]]

weights_sym_steep = [[1.0,0.9,0.1,0.0,0.0], 
                     [0.9,1.0,0.9,0.1,0.0], 
                     [0.1,0.9,1.0,0.9,0.1], 
                     [0.0,0.1,0.9,1.0,0.9], 
                     [0.0,0.0,0.1,0.9,1.0]]

weights_manual = [[1.0,0.9,0.2,0.1,0.1], 
                  [0.9,1.0,0.9,0.2,0.2], 
                  [0.2,0.5,1.0,0.9,0.2], 
                  [0.0,0.1,0.9,1.0,0.9], 
                  [0.0,0.1,0.2,0.9,1.0]]

## MAIN: CALCULATE THE FIGURES OF MERIT ##
# Confusion matrix follows the convention here
# https://i.stack.imgur.com/a3hnS.png. The element in the ith row and jth column
# is the the number of elements with ground truth j which were classified as i 
# by the model.

# Confusion matrix must have num_classes rows and columns. All elements should
# be non-negative integers.

num_classes = 5
conf_matrix = confusion_matrix_A.to_numpy(dtype=int)
weights = weights_sym_steep

print(conf_matrix)
print(weights)
print(sum([sum(r) for r in conf_matrix]))
for i in range(num_classes):
  print("Gen. Precision for label", i)
  generalized_precision(conf_matrix, weights, i)
  print("Gen. Recall for label", i)
  generalized_recall(conf_matrix, weights, i)
  print("Overall Gen. Accuracy:", generalized_accuracy(conf_matrix, weights))