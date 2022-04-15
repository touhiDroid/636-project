from types import SimpleNamespace

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


num_steps = 10000  # play with different values
summary_freq = 200
n_test_log = 10
n_outputs = 10
batch_size = 100  # play with different values for batch size


# Classes from previous exercises
class DenseLayer(object):
    def __init__(self, n_inputs, n_units, afunc=None, w_stddev=0.01):
        '''Define the parameters of the layer'''
        self.w = tf.Variable(
            tf.random.truncated_normal([n_inputs, n_units], stddev=w_stddev),
            name='w')
        self.b = tf.Variable(
            tf.zeros([n_units]),
            name='b')
        self.afunc = afunc

    def trainable_variables(self):
        '''return trainable variables'''
        return [self.w, self.b]

    def __call__(self, x):
        '''Layer function definition'''
        y = tf.matmul(x, self.w) + self.b
        if self.afunc is not None:
            y = self.afunc(y)
        return y


class LogisticReg(object):
    def __init__(self, layers):
        self.layers = layers

    def trainable_variables(self):
        return [var for layer in self.layers
                for var in layer.trainable_variables()]

    def __call__(self, x, logits=False):
        '''call layers and apply softmax if logits=False'''
        # compute layers
        output = x
        for layer in self.layers:
            output = layer(output)
        # apply softmax if logits is false
        # use logits=True for training
        if not logits:
            output = tf.nn.softmax(output)
        return output


# load mnist dataset with labels encoded as one-hot vectors
class Dataset(object):
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.epochs = 0

    def shuffle(self):
        perm = np.arange(self.data[0].shape[0])
        np.random.shuffle(perm)
        self.data = tuple(datai[perm] for datai in self.data)

    def next_batch(self, batch_size):
        start = self.index
        end = self.index + batch_size
        if end > self.data[0].shape[0]:
            self.epochs += 1
            self.shuffle()
            self.index, start = 0, 0
            end = batch_size
        self.index = end
        return tuple(datai[start:end, ...] for datai in self.data)


def load_mnist():
    def preprocess(data, labels, num_classes):
        # flatten images
        data = data.astype(np.float32) / 255.0
        data = np.reshape(data, [data.shape[0], -1])
        # one hot encoding
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        return data, labels_one_hot

    train, test = tf.keras.datasets.mnist.load_data()
    train = preprocess(train[0], train[1], 10)
    test = preprocess(test[0], test[1], 10)
    return SimpleNamespace(
        train=Dataset(train),
        test=Dataset(test))


class ConvLayer(object):
    def __init__(self, input_maps, output_maps, filter_size,
                 pool_size, afunc=None):
        '''
  Convolution layer with VALID padding and pooling layer.

  input_maps: number of input maps.
  output_maps: number of output maps.
  filter_size: list/tuple with the size of the kernel filter.
  pool_size: list/tuple with the size of the pool filter.
  afunc: activation function.
  '''
        self.w = tf.Variable(tf.random.truncated_normal(
            shape=[filter_size[0], filter_size[1],
                   input_maps, output_maps],
            stddev=0.1), name='w')
        self.b = tf.Variable(tf.random.truncated_normal(
            shape=[output_maps],
            stddev=0.1), name='b')
        self.pool_size = pool_size
        self.afunc = afunc

    def trainable_variables(self):
        '''return trainable variables'''
        return [self.w, self.b]

    def __call__(self, x):
        out = tf.nn.conv2d(
            x, self.w, strides=[1, 1, 1, 1],
            padding='VALID')
        out = out + self.b
        if self.afunc is not None:
            out = self.afunc(out)
        out = tf.nn.max_pool(
            out, ksize=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.pool_size[0], self.pool_size[1], 1],
            padding='VALID')
        return out


class ReshapeLayer(object):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def trainable_variables(self):
        '''return trainable variables'''
        return []

    def __call__(self, x):
        return tf.reshape(x, self.output_shape)


def loss_fn(logits, labels, weights):
    '''compute softmax cross entory'''
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels))
    # add an l2 regularization loss
    reg = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights])
    return error + 0.0001 * reg


def train_step(labels, inputs):
    '''run a single step of gradient descent'''
    # compute gradients
    with tf.GradientTape() as tape:
        logits = model(inputs, logits=True)
        loss = loss_fn(logits, labels, weights)
    # apply grandients to optimizer
    gradients = tape.gradient(loss, model.trainable_variables())
    optimizer.apply_gradients(zip(gradients, model.trainable_variables()))
    return loss.numpy(), model(inputs).numpy()


def accuracy(predictions, labels):
    if n_outputs == 1:
        return (100.0 * np.sum(np.greater(predictions, 0.5) == np.greater(labels, 0.5)) / predictions.shape[0])
    else:
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


if __name__ == '__main__':
    mnist = load_mnist()
    # define model
    model = LogisticReg([
        ConvLayer(input_maps=1, output_maps=32, filter_size=(5, 5), pool_size=(2, 2), afunc=tf.nn.relu),
        ConvLayer(input_maps=32, output_maps=64, filter_size=(5, 5), pool_size=(2, 2), afunc=tf.nn.relu),
        ReshapeLayer(output_shape=[-1, 4 * 4 * 64]),  # HINT: where does 4*4*64 comes from?
        # use the architecture given in the
        # assignment to figure this out !!!
        # Answer: The width of the output of
        # first conv2d is calculated using (W−F+2P)/S+1
        # (28-5+2*0)/1+1= 24 and then the output of
        # maxpool is 24/2 = 12, then the output of second
        # conv2d is (12-5+2*0)/1+1=8, then the output of
        # maxpool is 8/2 = 4 and the same happens for the
        # height so the outcome of second ConvLayer
        # is W*H*output_channel = 4*4*64
        DenseLayer(4 * 4 * 64, 256, tf.nn.relu),
        DenseLayer(256, 10)
    ])  # COMPLETE
    weights = [layer.w for layer in model.layers if hasattr(layer, 'w')]

    # define optimizer.
    optimizer = tf.keras.optimizers.Adam(1e-3)

    mean_loss = 0
    train_accuracy = 0
    for step in range(num_steps):
        # Get next batch of 100 images
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        # The images returned by the function are formated in a matrix,
        # where each row represents an image. Hence, we must reshape such
        # matrix to convert the vector-representation of the images to
        # standard 28 by 28 grey images.
        batch_X = np.reshape(batch_X, [-1, 28, 28, 1])

        # Call the optimizer to perform one step of the training
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
            for i in range(n_test_log):
                batch_X_test, batch_y_test = mnist.test.next_batch(batch_size)
                batch_X_test = np.reshape(batch_X_test, [-1, 28, 28, 1])
                pred = model(batch_X_test)
                test_accuracy += accuracy(pred, batch_y_test)
            test_accuracy = test_accuracy / n_test_log
            # ------------------------------- #
            print(step, ', train:', train_accuracy, ' | test:', test_accuracy, ' | loss:', mean_loss / summary_freq)
            mean_loss = 0
            train_accuracy = 0

    # Acquire one sample from the mnist dataset
    test_sample_x, test_sample_y = mnist.test.next_batch(1)

    # Evaluate the training model on test_sample_x
    # and compare it with the actual label test_sample_y
    test_sample_x = np.reshape(test_sample_x, [-1, 28, 28, 1])
    pred = model(test_sample_x)
    print('Number:', np.argmax(test_sample_y))
    print('Prediction by the model:', np.argmax(pred))
    # ------------------------------- #

    # Plot
    # plt.imshow(np.squeeze(test_sample_x))
