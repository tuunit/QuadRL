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

class Config:
    TRAJECTORIES_N = 10#512
    INITIAL_N = 10#512
    BRANCHES_N = 10#1024
    INITIAL_LENGTH = 10#512
    BRANCH_LENGTH = int(10)#512)
    TAIL_STEPS = 2#128
    NOISE_DEPTH = 1
    VALUE_ITERATIONS = 300
    VALUE_LOSS_LIMIT = 0.0001
    DISCOUNT_VALUE = 0.99
    TIME_STEP = 0.01
    ACTION_BIAS = .00
    ACTION_SCALE = 4.
    NOISE_COV = np.matrix([[.0062, 0., 0., 0.],
                           [0., .0062, 0., 0.],
                           [0., 0., .0062, 0.],
                           [0., 0., 0., .0062]], dtype=np.float64)
    CHOLESKY_COV = np.linalg.cholesky(NOISE_COV)
    ANGULAR_VEL_NORM = 1/2.
    LINEAR_VEL_NORM = 1/2.
    POSITION_NORM = 1/2.

    POLICY_SHAPE = [18, 128, 128, 4]
    VALUE_SHAPE  = [18, 128, 128, 1]

class Utils:
    def noise(num):
        return np.array((np.random.normal(0, 1, size=(num, 4))) * np.float64(Config.CHOLESKY_COV))

    def normalize_states(states):
        n_state = np.array(states)
        n_state[:, 9:12]  = n_state[:, 9:12] * Config.POSITION_NORM
        n_state[:, 12:15] = n_state[:, 12:15] * Config.ANGULAR_VEL_NORM
        n_state[:, 15:18] = n_state[:, 15:18] * Config.LINEAR_VEL_NORM
        return n_state
    
    def forward(sess, network, states):
        prediction = network.model()
        states = Utils.normalize_states(states)
    
        return sess.run(prediction, feed_dict={network.input: states})

    def value_function_vectorized(costs, terminal_value):
        values = np.zeros(len(costs))
        values[-1] = terminal_value
    
        for i in range(len(costs)-2, -1, -1):
            values[i] = costs[i] + Config.DISCOUNT_VALUE * values[i+1]
    
        return values



