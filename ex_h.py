import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("/home/matt/Documents/3f8"))
from inference_functions import *

# Check Arguments
if len(sys.argv) != 2:
    print("Usage: {} l".format(sys.argv[0]))
    sys.exit(1)

# Read hyper parameter
l = float(sys.argv[1])

# Parameters
if(l==0.01):
    eta = 0.014          # Learning rate
if (l==1):
    eta = 0.001
else:
    eta = 0.01
steps = 1000             # Training steps
w = np.ones(801)         # Weights
ll = np.zeros(steps)     # Average LL Evolution
num_test = 200
num_train = 800

# Load training data
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')

#Load test data
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')

# Expand using RBFs
X_train_expanded = expand_inputs(l, X_train, X_train)
X_train_expanded = np.insert(X_train_expanded,800,1,1)
X_test_expanded = expand_inputs(l, X_test, X_train)
X_test_expanded = np.insert(X_test_expanded,800,1,1)

# Train Weights
for i in range (0, steps):
    
    # Compute dL(w)/dw
    dw = np.dot(np.transpose(X_train_expanded),y_train-logistic(np.dot(X_train_expanded,w)))
    
    # Perform Gradient Ascent
    w = w + eta*dw
    
    # Update LL Evolution
    ll[i] = compute_average_ll(X_train_expanded,y_train,w)
    
# Plot LL evolution
print("FINAL AVG TRAIN LL =", ll[steps-1])
print("FINAL AVG TEST LL =", compute_average_ll(X_test_expanded, y_test, w))

# Plot Classification Regions
plot_expanded_predictive_distribution(X_train,X_train,y_train,w,l)
plot_expanded_predictive_distribution(X_test,X_train,y_test,w,l)

# Calculate probabilities
train_predictions = predict_for_plot(X_train, w)
test_predictions = predict_for_plot(X_test, w)

#print("Training Predictions:", train_predictions)
#print("Testing Predictions:", test_predictions)

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

train_confusion[0][0] = true_negatives
train_confusion[0][1] = false_negatives
train_confusion[1][0] = false_positives
train_confusion[1][1] = true_positives
train_confusion = train_confusion/num_train
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

test_confusion[0][0] = true_negatives
test_confusion[0][1] = false_negatives
test_confusion[1][0] = false_positives
test_confusion[1][1] = true_positives
test_confusion = test_confusion/num_test
print("Testing Data Confusion Matrix:")
print(test_confusion)
