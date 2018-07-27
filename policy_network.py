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

            # placeholder for optimizing of formula 5
            self.states_f_l = tf.placeholder('int32')
            self.states_p_l = tf.placeholder('int32')

            self.value_f = tf.placeholder('float')
            self.value_p = tf.placeholder('float')

            self.junction_state = tf.placeholder('float', [18])
            self.action_noise = tf.placeholder('float', [4])

            action_pred = self.model()

            # jacobian action wrt parameters
            J = tf.stack([tf.concat([tf.reshape(tf.gradients(action_pred[:, idx].__getitem__(self.states_f_l), param), [1, -1])
                                                for param in variables], 1) for idx in range(4)],
                                                axis=1, name='jac_Action_wrt_Param')[0]

            # svd for hessian pseudo inverse
            s_, U_, V_ = tf.svd(tf.matmul(tf.cholesky(tf.matrix_inverse(NOISE_COV)), J))
            h00_pinv = tf.matmul(tf.matmul(V_, tf.matrix_diag(tf.square(1/s_))), tf.transpose(V_))


            # Calculate cost function for states_f_l+states_p_l current position but with noise as action
            position = self.junction_state[9:12]
            angular = self.junction_state[12:15]
            linear = self.junction_state[15:18]
            
            position = 4 * 10**(-3) * tf.norm(position)
            angular = 3 * 10**(-4) * tf.norm(angular)
            linear = 5 * 10**(-4) * tf.norm(linear)
            action = (2/3.) * 10**(-5) * tf.norm(self.action_noise)

            rf = position + action + angular + linear

            # create value function for off policy
            iterf = tf.constant(0, dtype=tf.int32)
            vf = tf.constant(0, dtype=tf.float32)
            cf = lambda i, vf: tf.less(i, self.states_f_l)

            vf_w = tf.while_loop(cf, 
                    self.gen_body(self.input[:self.states_f_l],
                        action_pred, 
                        0), 
                    [iterf, vf])[1]
            vf_w += DISCOUNT_VALUE**tf.cast(self.states_f_l, dtype='float') * self.value_f
           
            # create value function for on policy
            iterp = tf.constant(0, dtype=tf.int32)
            vp = tf.constant(0, dtype=tf.float32)
            cp = lambda i, vp: tf.less(i, self.states_p_l)

            vp_w = tf.while_loop(cp, 
                    self.gen_body(self.input[self.states_f_l:],
                        action_pred, 
                        self.states_f_l), 
                    [iterp, vp])[1]

            vp_w += DISCOUNT_VALUE**tf.cast(self.states_p_l, dtype='float') * self.value_p

            # sum up all itermediate results
            self.A = rf + DISCOUNT_VALUE * vf_w - vp_w

            # calculate gradients of A to and multiply with J (gradients of a to param)
            grads = tf.gradients(self.A, action_pred)[0].__getitem__(self.states_f_l)
            gk = tf.matmul([grads], J)

            # multiply gk with hessian pseudo inverse
            nk = tf.matmul(gk, h00_pinv)
            self.train_op = nk
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)

    # value function for gradient calculation in formula 5
    def gen_body(self, states, actions, offset):
        def body(i, v):
            tmp = DISCOUNT_VALUE**tf.cast(i, dtype='float') 
            position = 4 * 10**(-3) * tf.norm(states[i][9:12])
            action = (2 / 3.) * 10 ** (-5) * tf.norm(actions[i+offset])
            linear = 3 * 10**(-4) * tf.norm(states[i][12:15]) 
            angular = 5 * 10**(-4) * tf.norm(states[i][15:18]) 
            cost = position + action + linear + angular
            tmp = tmp * cost
            v = v + tmp
            v.set_shape(tf.TensorShape([]))
            return i+1, v
        return body
