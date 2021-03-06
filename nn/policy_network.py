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
import tensorflow as tf
from .neural_network import NeuralNet

class PolicyNet(NeuralNet):
    """ Inherited class of NeuralNet for the policy optimization
    """
    def __init__(self, shape, noise_cov, graph=None):
        NeuralNet.__init__(self, shape, graph)
        with self.graph.as_default():
            # initiliaze policy network variables
            variables = tf.trainable_variables()

            self.action_grads = tf.placeholder('float64', [1, 4])
            action_pred = self.model()

            # jacobian action wrt parameters
            J = tf.stack([tf.concat([tf.reshape(tf.gradients(action_pred[0, idx], param), [1, -1])
                                                for param in variables], 1) for idx in range(4)],
                                                axis=1, name='jac_Action_wrt_Param')[0]

            gk = tf.matmul(self.action_grads, J)

            # svd for hessian pseudo inverse
            s_, U_, V_ = tf.svd(tf.matmul(tf.transpose(tf.cholesky(tf.matrix_inverse(noise_cov))), J))
            h00_pinv = tf.matmul(tf.matmul(V_, tf.matrix_diag(tf.square(1/s_))), tf.transpose(V_))

            # multiply gk with hessian pseudo inverse
            nk = tf.matmul(h00_pinv, tf.transpose(gk))
            self.train_op = tf.transpose(nk)
            self.beta = tf.sqrt(tf.constant(14000., dtype=tf.float64) / tf.matmul(gk, nk))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.)
