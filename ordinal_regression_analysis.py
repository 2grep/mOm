import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

df = pd.read_csv('./Steiner2020_assisted.csv')
print(df.nunique())
df.head(15)
confusion_matrix_A = pd.crosstab(df['GU_majority_Ground_truth'], df['B'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix_A)

## FUNCTIONS ##
# David wrote these, originally here: 
# https://colab.research.google.com/drive/10By9_PZLvDY9EAa-n_tt8RGvSfoaQO8x

def generalized_accuracy(conf_matrix, weights):
    '''
    Generalized accuracy takes in a confusion matrix and returns a measure of the overall predictive value of a given classifier.
    '''
    score = 0
    num_elements = sum([sum(r) for r in conf_matrix])
    if num_elements == 0:
      raise("No elements")
      
    for i, row in enumerate(conf_matrix):
        for j, x in enumerate(row):
            score += x * weights[j][i]

    return score / num_elements

def error_severity_index(conf_matrix, esi_weights):
    '''
    Error severity index takes in a confusion matrix and returns a measure of the overall predictive value of a given classifier. TODO: check that the conf_matrix is of the same shape as the esi_weights
    '''
    score = 0
    num_elements = sum([sum(r) for r in conf_matrix])
    if num_elements == 0:
      raise("No elements")
      
    for i, row in enumerate(conf_matrix):
        for j, x in enumerate(row):
            score += x * esi_weights[j][i]

    return score / num_elements

def generalized_recall(conf_matrix, weights, label_num):
    '''
    Generalized recall is computed for a specific label. For a given label, it computes a measure of all the label's actual instances which were corretly labelled.
    '''
    col = [row[label_num] for row in conf_matrix]

    if sum(col) == 0:
        raise("No elements with that ground truth label")

    score = 0
    show_work = "("
    for i, x in enumerate(col):
        score += x * weights[label_num][i]

        # to print out the work
        show_work += ("{}⋅{:.2f}").format(x, weights[i][label_num])
        if i != len(col) - 1:
            show_work += (" + ")
        else:
            show_work += (") / ({}) = ").format(sum(col))
        # to print out the work

    gen_recall = score / sum(col)
    #print(show_work, gen_recall)
    print(show_work, gen_recall)

    return gen_recall

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
        show_work += ("{}⋅{:.2f}").format(x, weights[label_num][i])
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

esi_weights = [[0,1,3,5,5], 
               [0,0,1,3,5], 
               [1,0,0,1,3], 
               [2,1,0,0,1], 
               [2,2,1,0,0]]

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
print("Error severity index:", error_severity_index(conf_matrix, esi_weights))