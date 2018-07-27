from neural_network import NeuralNet
import tensorflow as tf

class ValueNet(NeuralNet):
    """ Inherited class of NeuralNet for the value optimization
    """
    def __init__(self, shape, graph=None):
        NeuralNet.__init__(self, shape, graph)
        with self.graph.as_default():
            self.loss = tf.reduce_mean(tf.losses.huber_loss(predictions=self.model(),labels=self.output))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
            self.train_op = self.optimizer.minimize(self.loss)
