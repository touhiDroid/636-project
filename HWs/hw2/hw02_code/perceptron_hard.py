# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Contains: Perceptron Learning Rule with Hard Threshold
# Name: perceptron_hard.py
# Course Instructor: Milos Manic
# Provided by: Course Instructor
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0.5


def print_data(n, p, net, error, learned, weights, writefile):
    print('ite= {} p= {} net= {} err= {} lrn= {}\nweights: {}'.format(
        n, p, round(net, 2), round(error, 3), round(learned, 3),
        ' '.join(str(round(weight, 2)) for weight in weights)))
    writefile.write('ite= {} p= {} net= {} err= {} lrn= {}\nweights: {}'.format(
        n, p, round(net, 2), round(error, 3), round(learned, 3),
        ' '.join(str(round(weight, 2)) for weight in weights)) + '\n')


if __name__ == '__main__':
    # Open a new file for writing output.
    output = 'out_perc_hard.txt'
    with open(output, 'w') as write_file:
        iterations = 100        # Number of training cycles
        num_patterns = 2        # Number of patterns
        num_inputs = 3          # Number of augmented inputs
        alpha = 0.3             # Learning constant
        weights = [1, 3, -3]    # List of weights
        patterns = [[1, 2, 1], [2, 1, 1]]   # Patterns as a 2-dimensional list
        desired_out = [-1, 1]   # Desired output as a 1-dimensional list

        # For each iteration
        for n in range(iterations):
            total_error = 0     # Total error
            # Perceptron's predicted output for each pattern
            predicted_out = [0 for _ in range(num_patterns)]

            # For each pattern
            for p in range(num_patterns):
                # Net of weights * inputs
                net = sum(weight * pattern 
                          for weight, pattern in zip(weights, patterns[p]))

                # Use output function
                predicted_out[p] = sign(net)
                # Calculating error
                error = desired_out[p] - predicted_out[p]
                total_error += error**2
                # Learning coefficient
                learned = alpha * error

                # Print data to output file & standard out
                print_data(n, p, net, error, learned, weights, write_file)

                # Update weights
                weights = [weight + learned * pattern 
                           for weight, pattern in zip(weights, patterns[p])]
            print('TE= ', round(total_error, 6))
            write_file.write('TE= ' + str(round(total_error, 6)) + '\n')

            # Exit loop if error is small
            if total_error < 0.001:
                break

    # Wait for user response
    input()
