from math import exp

import numpy


def sigmoid(x):
    k = 1
    return 1 / (1 + exp(-1 * k * x))


def delta(x):
    k = 1
    s = sigmoid(x)
    return k * s * (1 - s)


def sigmoid_bipolar(x):
    k = 1
    return 2 / (1 + exp(-2 * k * x)) - 1


# noinspection PyShadowingNames
def print_data(n, p, net, error, learned, weights, writefile):
    print('ite= {} p= {} net= {} err= {} lrn= {} weights: {}'.format(
        n, p, round(net, 2), round(error, 3), round(learned, 3),
        ' '.join(str(round(weight, 2)) for weight in weights)))
    writefile.write('ite= {} p= {} net= {} err= {} lrn= {} weights: {}'.format(
        n, p, round(net, 2), round(error, 3), round(learned, 3),
        ' '.join(str(round(weight, 2)) for weight in weights)) + '\n')


if __name__ == '__main__':
    # Open a new file for writing output.
    output = 'H213_delta.txt'
    with open(output, 'w') as write_file:
        iterations = 100  # Number of training cycles
        num_patterns = 8  # Number of patterns
        num_inputs = 4  # Number of augmented inputs
        alpha = 0.1  # Learning constant
        weights = [1, 1, 1, 1]  # List of weights
        patterns = [  # Patterns as a 2-dimensional list
            [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1],  # Each item = [A, B, C, Bias]
            [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
        desired_out = [0, 0, 0, 0, 0, 1, 0, 1]  # Desired output as a 1-dimensional list

        # For different learning constants: alpha = 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, ..., 19.9
        lr_ite_csv_file = open('lr_ite_delta.csv', 'w')
        for alpha in numpy.arange(0.1, 20, 0.2):
            # For each iteration
            for n in range(iterations):
                total_error = 0  # Total error
                # Perceptron's predicted output for each pattern
                predicted_out = [0.0 for _ in range(num_patterns)]

                # For each pattern
                for p in range(num_patterns):
                    # Net of weights * inputs
                    net = sum(weight * pattern for weight, pattern in zip(weights, patterns[p]))

                    # Use output function
                    predicted_out[p] = sigmoid(net)
                    # Calculating error
                    error = desired_out[p] - predicted_out[p]
                    total_error += error ** 2
                    # Learning coefficient
                    learned = alpha * error * delta(net)  # Apply Delta ?

                    # Print data to output file & standard out
                    print_data(n, p, net, error, learned, weights, write_file)

                    # Update weights
                    weights = [weight + learned * pattern
                               for weight, pattern in zip(weights, patterns[p])]
                print('TE= ', round(total_error, 6))
                write_file.write('TE= ' + str(round(total_error, 6)) + '\n')

                # Exit loop if error is small
                if total_error < 0.01:
                    break
            n += 1  # removing the 0-index effect from n to get the total number of iterations performed
            lr_report_str = '~~~~~~~~END: Learning Constant, alpha = {:.3f}\n' \
                            '\tTE={:.2f}\t\tWeight=[{:s}]\t\tIterations={:d}' \
                            '\n~~~~~~~~\n\n'.format(alpha, total_error,
                                                    ', '.join(str(round(weight, 2)) for weight in weights), n)
            lr_ite_csv_file.write(str(round(alpha, 3)) + ',' + str(total_error) + ',' + str(n) + ',' +
                                  ', '.join(str(round(weight, 2)) for weight in weights) + '\n')
            print(lr_report_str)
            write_file.write(lr_report_str)
        lr_ite_csv_file.close()
    # Wait for user response
    input()
