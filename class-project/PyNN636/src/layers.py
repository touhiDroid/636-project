import tensorflow as tf


# Classes from previous exercises
class DenseLayer(object):
    def __init__(self, n_inputs, n_units, afunc=None, w_stddev=0.01):
        """Define the parameters of the layer"""
        self.w = tf.Variable(
            tf.random.truncated_normal([n_inputs, n_units], stddev=w_stddev),
            name='w')
        self.b = tf.Variable(tf.zeros([n_units]), name='b')
        self.afunc = afunc

    def trainable_variables(self):
        """return trainable variables"""
        return [self.w, self.b]

    def __call__(self, x):
        """Layer function definition"""
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
        """call layers and apply softmax if logits=False"""
        # compute layers
        output = x
        for layer in self.layers:
            output = layer(output)
        # apply softmax if logits is false
        # use logits=True for training
        if not logits:
            output = tf.nn.softmax(output)
        return output


class ConvLayer(object):
    def __init__(self, input_maps, output_maps, filter_size,
                 pool_size, afunc=None):
        """
          Convolution layer with VALID padding and pooling layer.

          input_maps: number of input maps.
          output_maps: number of output maps.
          filter_size: list/tuple with the size of the kernel filter.
          pool_size: list/tuple with the size of the pool filter.
          afunc: activation function.
        """
        self.w = tf.Variable(tf.random.truncated_normal(
            shape=[filter_size[0], filter_size[1], input_maps, output_maps],
            stddev=0.1), name='w')
        self.b = tf.Variable(tf.random.truncated_normal(shape=[output_maps], stddev=0.1), name='b')
        self.pool_size = pool_size
        self.afunc = afunc

    def trainable_variables(self):
        """return trainable variables"""
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

    # noinspection PyMethodMayBeStatic
    def trainable_variables(self):
        """return trainable variables"""
        return []

    def __call__(self, x):
        return tf.reshape(x, self.output_shape)
