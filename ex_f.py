import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("/home/matt/Documents/3f8"))
from inference_functions import *

## Parameters
eta = 0.0005                # Learning rate
steps = 100                 # Training steps
w = np.ones(3)              # Weights
ll_train = np.zeros(800)
ll_test = np.zeros(200)
num_test = 200
num_train = 800

# Load training data
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')

# Append 1's to X's
X_train = np.insert(X_train,2,1,1)
X_test = np.insert(X_test,2,1,1)

# Train Weights
for i in range (0, steps):
    
    # Compute dL(w)/dw
    dw = np.dot(np.transpose(X_train),y_train-logistic(np.dot(X_train,w)))
    
    # Perform Gradient Ascent
    w = w + eta*dw

# Reload X datasets
X_train = np.loadtxt('X_train.txt')
X_test = np.loadtxt('X_test.txt')

# Calculate probabilities
train_predictions = predict_for_plot(X_train, w)
test_predictions = predict_for_plot(X_test, w)

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0
train_confusion = np.zeros((2,2))
test_confusion = np.zeros((2,2))

# Trainng Data Error Analysis
for k in range (0, 800):

    if(train_predictions[k] > 0.5):
        
        # Classified Y = 1
        if(y_train[k] == 1.0):
            true_positives = true_positives+1
        else:
            false_positives = false_positives+1
    else:
  
        # Classified Y = 0        
        if(y_train[k] == 0.0):
            true_negatives = true_negatives+1
        else:
            false_negatives = false_negatives+1     

train_confusion[0][0] = true_negatives/(true_negatives+false_negatives)
train_confusion[0][1] = false_negatives/(true_negatives+false_negatives)
train_confusion[1][0] = false_positives/(true_positives+false_positives)
train_confusion[1][1] = true_positives/(true_positives+false_positives)
print("Training Data Confusion Matrix:")
print(train_confusion)


true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# Test Data Error Analysis
for m in range (0, 200):

    if(test_predictions[m] > 0.5):
        
        # Classified Y = 1
        if(y_test[m] == 1.0):
            true_positives = true_positives+1
        else:
            false_positives = false_positives+1
    else:
  
        # Classified Y = 0        
        if(y_test[m] == 0.0):
            true_negatives = true_negatives+1
        else:
            false_negatives = false_negatives+1     

test_confusion[0][0] = true_negatives/(true_negatives+false_negatives)
test_confusion[0][1] = false_negatives/(true_negatives+false_negatives)
test_confusion[1][0] = false_positives/(true_positives+false_positives)
test_confusion[1][1] = true_positives/(true_positives+false_positives)
print("Testing Data Confusion Matrix:")
print(test_confusion)
