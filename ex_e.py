import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("/home/matt/Documents/3f8"))
from inference_functions import *

## Parameters
eta = 0.0005            # Learning rate
steps = 100             # Training steps
w = np.ones(3)          # Weights
ll = np.zeros(steps)    # Average LL Evolution

# Load training data
X = np.loadtxt('X_train.txt')
y = np.loadtxt('y_train.txt')

# Append 1's to X
X = np.insert(X,2,1,1)

# Train Weights
for i in range (0, steps):
    
    # Compute dL(w)/dw
    dw = np.dot(np.transpose(X),y-logistic(np.dot(X,w)))
    
    # Perform Gradient Ascent
    w = w + eta*dw
    
    # Update LL Evolution
    ll[i] = compute_average_ll(X,y,w)

# Display Weights
print("Final Weights:", w)
    
# Plot LL evolution
plot_ll(ll)
print("FINAL AVG TRAIN LL =", ll[steps-1])

# Plot Classification Regions
X = np.loadtxt('X_train.txt')
plot_predictive_distribution(X,y,w)

# Compare test data
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')
X_test_3 = np.insert(X_test,2,1,1)
print("FINAL AVG TEST LL =", compute_average_ll(X_test_3,y_test,w))
plot_predictive_distribution(X_test,y_test,w)
