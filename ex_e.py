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
ll_train = np.zeros(steps)  # Average LL Evolution
ll_test = np.zeros(steps)  

# Load training data
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')


# Append 1's to X
X_train = np.insert(X_train,2,1,1)
X_test = np.insert(X_test,2,1,1)

# Train Weights
for i in range (0, steps):
    
    # Compute dL(w)/dw
    dw = np.dot(np.transpose(X_train),y_train-logistic(np.dot(X_train,w)))
    
    # Perform Gradient Ascent
    w = w + eta*dw
    
    # Update LL Evolution
    ll_train[i] = compute_average_ll(X_train,y_train,w)
    ll_test[i] = compute_average_ll(X_test,y_test,w) 

# Display Weights
print("Final Weights:", w)
    
# Plot LL evolution
plot_2_ll(ll_test, ll_train)

print("FINAL AVG TRAIN LL =", ll_train[steps-1])
print("FINAL AVG TEST LL =", ll_test[steps-1])

# Plot Classification Regions
X_train = np.loadtxt('X_train.txt')
plot_predictive_distribution(X_train,y_train,w)

# Compare test data
X_test = np.loadtxt('X_test.txt')
plot_predictive_distribution(X_test,y_test,w)

# Visualise on whole dataset
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')
plot_predictive_distribution(X,y,w)
