import tensorflow as tf

from constants import *
from neural_network import NeuralNet

class PolicyNet(NeuralNet):
    """ Inherited class of NeuralNet for the policy optimization
    """
    def __init__(self, shape, graph=None):
        NeuralNet.__init__(self, shape, graph)
        with self.graph.as_default():
            # initiliaze policy network variables
            variables = tf.trainable_variables()

            self.action_grads = tf.placeholder('float32', [1, 4])
            action_pred = self.model()

            # jacobian action wrt parameters
            J = tf.stack([tf.concat([tf.reshape(tf.gradients(action_pred[0, idx], param), [1, -1])
                                                for param in variables], 1) for idx in range(4)],
                                                axis=1, name='jac_Action_wrt_Param')[0]

            gk = tf.matmul(self.action_grads, J)

            # svd for hessian pseudo inverse
            s_, U_, V_ = tf.svd(tf.matmul(tf.transpose(tf.cholesky(tf.matrix_inverse(NOISE_COV))), J))
            h00_pinv = tf.matmul(tf.matmul(V_, tf.matrix_diag(tf.square(1/s_))), tf.transpose(V_))

            # multiply gk with hessian pseudo inverse
            nk = tf.matmul(h00_pinv, tf.transpose(gk))
            self.train_op = gk
            self.beta = tf.sqrt(tf.constant(14000., dtype=tf.float32) / tf.matmul(gk, nk))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.)
