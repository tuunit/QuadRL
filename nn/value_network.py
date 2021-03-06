#########################################################
#                                                       #
#   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR   #
#  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR  #
#  %    %## #    CC        V    V  MM M  M MM RR    RR  #
#   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR    #
#   (%      %,   CC    CC   VVVV   MM      MM RR   RR   #
#     #%    %*    CCCCCC     VV    MM      MM RR    RR  #
#    .%    %/                                           #
#       (%.      Computer Vision & Mixed Reality Group  #
#                                                       #
#########################################################
#   @copyright    Hochschule RheinMain,                 #
#                 University of Applied Sciences        #
#      @author    Jan Larwig, Sohaib Zahid              #
#     @version    1.0.0                                 #
#        @date    08.10.2018                            #
#########################################################
import numpy as np
import tensorflow as tf
from .neural_network import NeuralNet

class ValueNet(NeuralNet):
    """ Inherited class of NeuralNet for the value optimization
    """
    def __init__(self, shape, graph=None):
        NeuralNet.__init__(self, shape, graph)
        with self.graph.as_default():
            self.loss = tf.losses.mean_squared_error(predictions=self.model(), labels=self.output)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            grad = self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)

            maxnorm = tf.constant(100, dtype =tf.float64)
            grads, variables = zip(*grad)
            grads, gradnorm = tf.clip_by_global_norm(grads, clip_norm=maxnorm)
            grad = zip(grads, variables)
            self.train_op = self.optimizer.apply_gradients(grad)

    def train(self, sess, value_batch, state_batch):
        value_batch = np.array(value_batch).reshape([-1, 1])
        _, c = sess.run([self.train_op, self.loss], 
                    feed_dict = {
                        self.input: state_batch,
                        self.output: value_batch
                    })
        return c
