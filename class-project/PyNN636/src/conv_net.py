import numpy as np
import tensorflow as tf

from layers import LogisticReg, ConvLayer, ReshapeLayer, DenseLayer

# data/csi.csv contains 33,10,009 = (90 * 90 * 36777.833) lines of CSI data, each with 90 CSI, so ~367 batches for W=100
CSI_WINDOW_SIZE = 100
num_steps = 100  # play with different values
summary_freq = 200
n_test_log = 10
n_outputs = 10
batch_size = 100  # play with different values for batch size


def loss_fn(logits, labels, weights_for_loss):
    """compute softmax cross entropy"""
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels))
    # add a l2 regularization loss
    reg = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights_for_loss])
    return error + 0.0001 * reg


def train_step(labels, inputs):
    """run a single step of gradient descent"""
    # compute gradients
    with tf.GradientTape() as tape:
        logits = model(inputs, logits=True)
        loss = loss_fn(logits, labels, weights)
    # apply gradients to optimizer
    gradients = tape.gradient(loss, model.trainable_variables())
    optimizer.apply_gradients(zip(gradients, model.trainable_variables()))
    return loss.numpy(), model(inputs).numpy()


def accuracy(predictions, labels):
    if n_outputs == 1:
        return 100.0 * np.sum(np.greater(predictions, 0.5) == np.greater(labels, 0.5)) / predictions.shape[0]
    else:
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def parse_csi_data():
    batch = 0
    with open("../data/csi.csv", 'r') as csiFile:
        labels = []
        csi = [[[]]]
        lc = 0
        for line in csiFile:
            vals = line.split(",")
            # TODO: labels should be an 2-D array of Wx12 size,
            #  each with probability [0,1] of the corresponding action of a batch among the W batches
            labels.append(vals[0])
            for j in range(1, len(vals)):
                csi[batch][lc][j - 1] = vals[j]
            lc += 1
            if lc >= CSI_WINDOW_SIZE:
                batch += 1
                lc = 0
        return csi, labels


if __name__ == '__main__':
    # define model
    model = LogisticReg([
        ConvLayer(input_maps=1, output_maps=32,
                  filter_size=(5, 5), pool_size=(2, 2), afunc=tf.nn.relu),
        ConvLayer(input_maps=3 * 30 * CSI_WINDOW_SIZE * 2, output_maps=3 * 30 * CSI_WINDOW_SIZE,
                  filter_size=(5, 5), pool_size=(2, 2), afunc=tf.nn.relu),
        ReshapeLayer(output_shape=[-1, 4 * 4 * 3 * 30 * CSI_WINDOW_SIZE]),
        DenseLayer(4 * 4 * 3 * 30 * CSI_WINDOW_SIZE, 540, tf.nn.relu),
        DenseLayer(540, 12)
    ])
    weights = [layer.w for layer in model.layers if hasattr(layer, 'w')]

    # define optimizer.
    optimizer = tf.keras.optimizers.Adam(1e-3)

    mean_loss = 0
    train_accuracy = 0
    for step in range(num_steps):
        # Get next batch of 100 images
        # TODO: Get all batches (W CSI-lines, each for a single activity) & then process 1-batch at a time
        batch_X, batch_y = parse_csi_data()  # return 2-D array as [W] x [3*30], which is the next same-labeled

        # Call the optimizer to perform one step of the training
        # for-loop
        l, train_pred = train_step(batch_y, batch_X)
        # Compute accuracy
        train_accuracy += accuracy(train_pred, batch_y)
        mean_loss += l

        if step % summary_freq == 0:
            # obtain train accuracy
            train_accuracy = train_accuracy / summary_freq

            # Evaluate the test accuracy on a series of mini-batches
            # extracted from the testing dataset.
            # Use mini-batches of around ~100 images
            test_accuracy = 0
            # for i in range(n_test_log):
            #     batch_X_test, batch_y_test = # mnist.test.next_batch(batch_size)
            #     batch_X_test = np.reshape(batch_X_test, [-1, 28, 28, 1])
            #     pred = model(batch_X_test)
            #     test_accuracy += accuracy(pred, batch_y_test)
            # test_accuracy = test_accuracy / n_test_log
            # ------------------------------- #
            print(step, ', train:', train_accuracy, ' | test:', test_accuracy, ' | loss:', mean_loss / summary_freq)
            mean_loss = 0
            train_accuracy = 0

    # Acquire one sample-window (100 CSI) from the dataset & test it

    # ------------------------------- #
