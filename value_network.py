from neural_network import NeuralNet
import tensorflow as tf

class ValueNet(NeuralNet):
    """ Inherited class of NeuralNet for the value optimization
    """
    def __init__(self, shape, graph=None):
        NeuralNet.__init__(self, shape, graph)
        with self.graph.as_default():
            self.loss = tf.losses.mean_squared_error(predictions=self.model(), labels=self.output)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            grad = self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)

            maxnorm = tf.constant(100, dtype =tf.float32)
            grads, variables = zip(*grad)
            grads, gradnorm = tf.clip_by_global_norm(grads, clip_norm=maxnorm)
            grad = zip(grads, variables)

            self.train_op = self.optimizer.apply_gradients(grad)
