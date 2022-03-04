import numpy
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    # Input/Output patterns
    X = [  # Patterns as a 2-dimensional list
        [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1],  # Each item = [A, B, C, Bias]
        [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
    y = [0, 0, 0, 0, 0, 1, 0, 1]

    lr_ite_csv_file = open('lr_ite_hw14_555.csv', 'w')
    for alpha in numpy.arange(0.1, 10, 0.1):
        # Create model object
        clf = MLPClassifier(hidden_layer_sizes=(5, 5, 5), random_state=5, verbose=True, alpha=alpha,
                            activation='tanh', learning_rate_init=alpha, learning_rate='constant')
        # Fit data onto the model
        clf.fit(X, y)

        # Make prediction on test dataset
        yPred = clf.predict([[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 0, 1], [1, 1, 1, 1]])
        print('prediction', yPred)

        # Calculate accuracy
        acc_score = accuracy_score([1, 0, 0, 1], yPred)
        print(acc_score)
        lr_ite_csv_file.write(str(round(alpha, 1)) + ','
                              + str(clf.n_iter_) + ',' + str(clf.loss_) + ',' + str(acc_score) + '\n')
    lr_ite_csv_file.close()
