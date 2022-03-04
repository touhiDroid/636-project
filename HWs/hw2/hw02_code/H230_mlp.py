import numpy as np
import pandas as pd
# Import accuracy score
from sklearn.metrics import accuracy_score

# Import MLPClassifier
from sklearn.neural_network import MLPClassifier

# Input/Output patterns
X = [[0., 0.], [1., 1.]]
y = [0, 1]

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(2, 2),
                    random_state=5,
                    verbose=True,
                    alpha=0.01,
                    activation='tanh',
                    learning_rate_init=0.01, learning_rate='constant')

# Fit data onto the model
clf.fit(X, y)

# Make prediction on test dataset
yPred = clf.predict([[2., 2.]])
print('prediction', yPred)

# Calculate accuracy
print(accuracy_score([1], yPred))
