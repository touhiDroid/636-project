import datetime
import itertools
import os
import random
import time

import numpy as np
import tensorflow as tf

from layers import LogisticReg, ConvLayer, ReshapeLayer, DenseLayer

# data/csi.csv contains 33,10,009 = (90 * 90 * 36777.833) lines of CSI data, each with 90 CSI, so ~367 batches for W=100
batch_size = 1024  # duration of the action = (WiFiWavesPerImage * batch_size) / Frequency = 90 * 32 / 256 = 11.25 seconds
num_steps = 25  # varied 15-100 steps
summary_freq = 5
n_test_log = 10
n_outputs = 12
FILTER_SIZE = 3
POOL_SIZE = 2
PADDING = 0
STRIDE = 1
FIRST_LAYER_FILTERS = 9  # Conv2D
FIRST_LAYER_OUT_COUNT = int(((90 - FILTER_SIZE + 2 * PADDING) / STRIDE + 1) / POOL_SIZE)
SECOND_LAYER_FILTERS = 18  # Conv2D
SECOND_LAYER_OUT_COUNT = int(((FIRST_LAYER_OUT_COUNT - FILTER_SIZE + 2 * PADDING) / STRIDE + 1) / POOL_SIZE)
THIRD_LAYER_OUT_MAP = int(SECOND_LAYER_OUT_COUNT * SECOND_LAYER_OUT_COUNT * SECOND_LAYER_FILTERS)  # ReShape-layer
FOURTH_LAYER_OUT_COUNT = n_outputs * 256  # Fully-Connected Dense-Layer
FIFTH_LAYER_OUT_COUNT = n_outputs
DATA_FILE_FULL_PATH = os.getcwd()[:os.getcwd().rfind('PyNN636')] + "PyNN636/data/csi.csv"


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
    start_time_csi_parsing = time.time()
    imgNo = 0
    with open(DATA_FILE_FULL_PATH, 'r') as csiFile:
        labels = np.zeros((36655, n_outputs), dtype=float)  # [[]]  # [imgNo] -- [Activity=12]
        csi = np.zeros((36655, 90, 90), dtype=float)  # [imgNo] -- [Line=90] -- [W=90]
        for lines90 in itertools.zip_longest(*[csiFile] * 90):  # reading 90-lines at a time to match 90-sub-carriers
            if len(lines90) != 90:
                break
            lc = 0
            activity = int(lines90[0].split(",")[0])
            lbl = np.zeros([n_outputs])
            lbl[activity - 1] = 1.0
            for line in lines90:
                if line is not None:
                    vals = line.split(",")
                if vals is None \
                        or len(vals) != 91 \
                        or int(vals[0]) != activity:
                    activity = -1
                    break
                # labels should be an 2-D array of Wx12 size,
                #  each with probability [0,1] of the corresponding action of a batch among the W batches
                labels[imgNo] = lbl
                csi[imgNo][lc] = [float(x) for x in vals[1:]]
                # for j in range(1, len(vals)):
                #     csi[imgNo][lc].append(vals[j])
                lc += 1
            if activity > 0:
                imgNo += 1
                percentage = "{:.2f}".format(100 * imgNo / (3310009 / 90))
                print(f"Parsed Image #{imgNo} ... Probable Completion: {percentage}%")
                # csi.append([])
                # labels.append([])
        print(f"Parsing Time Needed: {datetime.timedelta(seconds=time.time() - start_time_csi_parsing)}\n")
        # del csi[0][0]  # deleting empty list at [0][0], which was created during init. of csi=[[[]]] {better way?}
        return csi, labels, imgNo


if __name__ == '__main__':
    start_time = time.time()
    # define model
    model = LogisticReg([
        ConvLayer(input_maps=1, output_maps=FIRST_LAYER_FILTERS, filter_size=(FILTER_SIZE, FILTER_SIZE),
                  pool_size=(POOL_SIZE, POOL_SIZE), afunc=tf.nn.relu),
        ConvLayer(input_maps=FIRST_LAYER_FILTERS, output_maps=SECOND_LAYER_FILTERS,
                  filter_size=(FILTER_SIZE, FILTER_SIZE),
                  pool_size=(POOL_SIZE, POOL_SIZE), afunc=tf.nn.relu),
        ReshapeLayer(output_shape=[-1, THIRD_LAYER_OUT_MAP]),
        DenseLayer(THIRD_LAYER_OUT_MAP, FOURTH_LAYER_OUT_COUNT, tf.nn.relu),
        DenseLayer(FOURTH_LAYER_OUT_COUNT, FIFTH_LAYER_OUT_COUNT)
    ])
    weights = [layer.w for layer in model.layers if hasattr(layer, 'w')]

    # define optimizer.
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # Get all batches (W CSI-lines, each for a single activity) & then process 1-batch at a time
    # _X = 3-D array, dimension = [TOTAL_90x90_IMAGES] x [3*30] x [3*30]
    # _y = 2-D array, dimension = [TOTAL_90x90_IMAGES] x [12]  <= has only 1 or 0
    _X, _y, total_images = parse_csi_data()

    mean_loss = 0
    train_accuracy = 0
    for step in range(1, num_steps + 1):
        c = 0  # random.randint(0, len(_X) - batch_size)
        st = time.time()

        while c < total_images:
            batch_X = _X[c: c + batch_size]
            batch_y = _y[c: c + batch_size]
            c += batch_size

            batch_X = np.reshape(batch_X, [-1, 90, 90, 1])

            # Call the optimizer to perform one step of the training
            l, train_pred = train_step(batch_y, batch_X)
            # Compute accuracy
            acc = accuracy(train_pred, batch_y)
            train_accuracy += acc
            mean_loss += l
            print(c, '/', total_images, '\t(', "{:.2f}".format(100 * c / total_images),
                  '%) progress ==> Train Accuracy=', "{:.2f}".format(100 * train_accuracy / c),
                  '\tCurr. Acc.=', "{:.2f}".format(100 * acc / batch_size),
                  ',\tMean Loss=', mean_loss)

        print('Training Time ==> ', datetime.timedelta(seconds=time.time() - st))

        test_accuracy = 0
        if step % summary_freq == 0:
            # obtain train accuracy
            train_accuracy = train_accuracy / (summary_freq * total_images)

            # Evaluate the test accuracy on a series of mini-batches
            # extracted from the testing dataset.
            # Use mini-batches of around ~100 images
            test_accuracy = 0
            test_size = 100  # int(0.01 * total_images)
            for i in range(n_test_log):
                p = random.randint(0, len(_X) - test_size)
                batch_X_test = _X[p:p + test_size]
                batch_y_test = _y[p:p + test_size]
                batch_X_test = np.reshape(batch_X_test, [-1, 90, 90, 1])
                pred = model(batch_X_test)
                test_accuracy += accuracy(pred, batch_y_test)
                # TODO: Save predicted & original labels to generate confusion-matrix later
            test_accuracy = test_accuracy / n_test_log
            # ------------------------------- #
            print(step, '# train:', train_accuracy, ' | test:', test_accuracy, ' | loss:', mean_loss / summary_freq)
            mean_loss = 0
            train_accuracy = 0
        if test_accuracy > 0:
            print(step, '# train:', train_accuracy, ' | test:', test_accuracy, ' | loss:', mean_loss / summary_freq)
        print(step, '# Complete Iteration Time ==> ', datetime.timedelta(seconds=time.time() - st))
    print('Program Runtime ==> ', datetime.timedelta(seconds=time.time() - start_time))
