import tensorflow as tf
import numpy as np
from pyquaternion import Quaternion

from constants import *
from utils import *

from drone_interface_icarus import DroneInterface
from neural_network import NeuralNet
from policy_network import PolicyNet
from value_network import ValueNet

def gen_body(states, states_l, actions, value_factor, offset):
    def body(i, v):
        tmp = DISCOUNT_VALUE**tf.cast(i, dtype='float') 
        position = 4 * 10**(-3) * tf.norm(states[i][9:12])
        action = 2 * 10**(-4) * tf.norm(actions[i+offset]) 
        linear = 3 * 10**(-4) * tf.norm(states[i][12:15]) 
        angular = 5 * 10**(-4) * tf.norm(states[i][15:18]) 
        cost = position + action + linear + angular
        tmp = tmp + cost 
        v = v + tmp
        v.set_shape(tf.TensorShape([]))
        return i+1, v
    return body

sess = tf.Session()

policy_net = NeuralNet(shape=[18,64,64,4])
variables = tf.trainable_variables()

# placeholder for optimizing of formula 5
states_f_l = tf.placeholder('int32')
states_p_l = tf.placeholder('int32')

value_f = tf.placeholder('float')
value_p = tf.placeholder('float')

junction_state = tf.placeholder('float', [18])
action_noise = tf.placeholder('float', [4])

action_pred = policy_net.model()

# jacobian action wrt parameters
J = tf.stack([tf.concat([tf.reshape(tf.gradients(action_pred[:, idx].__getitem__(states_f_l), param), [1, -1])
                                    for param in variables], 1) for idx in range(4)],
                                    axis=1, name='jac_Action_wrt_Param')[0]


# svd for hessian pseudo inverse
s_, U_, V_ = tf.svd(tf.matmul(tf.cholesky(tf.matrix_inverse(NOISE_COV)), J))
h00_pinv = tf.matmul(tf.matmul(V_, tf.matrix_diag(tf.square(1/s_))), tf.transpose(V_))


# Calculate cost function for states_f_l+states_p_l current position but with noise as action
position = junction_state[9:12]
angular = junction_state[12:15]
linear = junction_state[15:18]

position = 4 * 10**(-3) * tf.norm(position)
angular = 3 * 10**(-4) * tf.norm(angular)
linear = 5 * 10**(-4) * tf.norm(linear)
action = 2 * 10**(-4) * tf.norm(action_noise)

rf = position + action + angular + linear

# create value function for off policy
iterf = tf.constant(0, dtype=tf.int32)
vf = tf.constant(0, dtype=tf.float32)
cf = lambda i, vf: tf.less(i, states_f_l)

vf_w = tf.while_loop(cf, 
        gen_body(policy_net.input[:states_f_l],
            states_f_l, 
            action_pred, 
            value_f, 0), 
        [iterf, vf])[1]
vf_w += DISCOUNT_VALUE**tf.cast(states_f_l, dtype='float') * value_f

# create value function for on policy
iterp = tf.constant(0, dtype=tf.int32)
vp = tf.constant(0, dtype=tf.float32)
cp = lambda i, vp: tf.less(i, states_p_l)

vp_w = tf.while_loop(cp, 
        gen_body(policy_net.input[states_f_l:],
            states_p_l,
            action_pred, 
            value_p, 
            states_f_l), 
        [iterp, vp])[1]

vp_w += DISCOUNT_VALUE**tf.cast(states_p_l, dtype='float') * value_p

# sum up all itermediate results
A = rf + DISCOUNT_VALUE * vf_w - vp_w

# calculate gradients of A to and multiply with J (gradients of a to param)
grads = tf.gradients(A, action_pred)[0].__getitem__(states_f_l)
gk = tf.matmul([grads], J)

# multiply gk with hessian pseudo inverse
nk = tf.matmul(gk, h00_pinv)
train_op = nk
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
sess.run(tf.global_variables_initializer())

value_graph = tf.Graph()
value_sess = tf.Session(graph=value_graph)
value_net = ValueNet(shape=[18,64,64,1], graph=value_graph)
with value_sess.as_default():
    with value_sess.graph.as_default():
        tf.global_variables_initializer().run()

with sess.as_default():
    param_update = 0
    states_f = np.random.rand(10,18)
    states_p = np.random.rand(10, 18)

    T_f = len(states_f)
    T_p = len(states_p)

    # Predict value / reward of both trajectories termination state
    with value_graph.as_default():
        val_f = value_net.model().eval(session=value_sess, feed_dict={value_net.input:[states_f[T_f-1]]})[0][0]
        val_p = value_net.model().eval(session=value_sess, feed_dict={value_net.input:[states_p[T_p-1]]})[0][0]

    states = np.concatenate((states_f, states_p))
    junction = states_p[0]
    noise = np.random.rand(4)*800 - 459

    # Feed the data to the optimization graph
    _A, nk, g, j, t = sess.run([A, train_op, grads, J, tf.gradients(A, [action_pred.__getitem__(T_f)])], \
            feed_dict={policy_net.input: states, \
            states_f_l: T_f, \
            states_p_l: T_p, \
            junction_state: junction, \
            action_noise: noise, \
            value_f: val_f, \
            value_p: val_p})
    print(t)
