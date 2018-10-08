import numpy as np
import tensorflow as tf

class NeuralNet:
    """ Generic neural network class to create fully connected networks of any size
    """
    def __init__(self, shape, graph=None):
        if graph:
            self.graph = graph
        else:
            self.graph = tf.get_default_graph()
        self.shape = shape
        self.hls = len(self.shape) - 2

        with self.graph.as_default():
            self.input = tf.placeholder('float64', [None, self.shape[0]])
            self.output = tf.placeholder('float64', [None, self.shape[-1]])

        self.weights = list()
        self.biases = list()
        self.layers = list()
        self.create_model()

    def create_model(self):
        with self.graph.as_default():
            if not self.weights or not self.biases:
                initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
                for i in range(len(self.shape)-2):
                    self.weights.append(tf.Variable(initializer([self.shape[i], self.shape[i+1]])))

                    self.biases.append(tf.Variable(initializer([self.shape[i+1]])))
                self.weights.append(tf.Variable(tf.random_uniform(shape=[self.shape[-2], self.shape[-1]], dtype=tf.float64, minval= -float(3e-3), maxval=float(3e-3))))

                self.biases.append(tf.Variable(tf.random_uniform(shape=[self.shape[-1]], dtype=tf.float64, minval= -float(3e-3), maxval=float(3e-3))))

            if not self.layers:
                for i, wb in enumerate(zip(self.weights, self.biases)):
                    if i == 0:
                        layer = tf.add(tf.matmul(self.input, wb[0]), wb[1])
                    else:
                        layer = tf.add(tf.matmul(self.layers[i-1], wb[0]), wb[1])

                    if i < self.hls:
                        layer = tf.nn.tanh(layer)

                    self.layers.append(layer)

    def model(self):
        return self.layers[-1]
