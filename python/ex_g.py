import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("/home/matt/Documents/3f8/python"))
from inference_functions import *

# Check Arguments
if len(sys.argv) != 2:
    print("Usage: {} l".format(sys.argv[0]))
    sys.exit(1)

# Read hyper parameter
l = float(sys.argv[1])

# Parameters
if(l==0.01):
    eta = 0.018          # Learning rate
if (l==1):
    eta = 0.001
else:
    eta = 0.01
steps = 1000             # Training steps
w = np.ones(801)         # Weights
ll = np.zeros(steps)     # Average LL Evolution

# Load training data
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')

#Load test data
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')

X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')

# Expand using RBFs
X_train_expanded = expand_inputs(l, X_train, X_train)
X_train_expanded = np.insert(X_train_expanded,800,1,1)
X_test_expanded = expand_inputs(l, X_test, X_train)
X_test_expanded = np.insert(X_test_expanded,800,1,1)
X_expanded = expand_inputs(l, X, X_train)
X_expanded = np.insert(X_expanded,800,1,1)

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
plot_ll(ll)

# Plot Classification Regions
plot_expanded_predictive_distribution(X_train,X_train,y_train,w,l)
plot_expanded_predictive_distribution(X_test,X_train,y_test,w,l)
plot_expanded_predictive_distribution(X,X_train,y,w,l)

