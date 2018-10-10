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
    TRAINING_EPOCHS = 512
    INITIAL_N = 512
    BRANCHES_N = 1024
    INITIAL_LENGTH = 512
    BRANCH_LENGTH = int(512)
    NOISE_DEPTH = 1
    VALUE_ITERATIONS = 300
    VALUE_LOSS_LIMIT = 0.0001
    DISCOUNT_VALUE = 0.99
    TIME_STEP = 0.01
    ACTION_BIAS = 0.0
    ACTION_SCALE = 2.
    NOISE_COV = np.matrix([[.2, 0., 0., 0.],
                           [0., .2, 0., 0.],
                           [0., 0., .2, 0.],
                           [0., 0., 0., .2]], dtype=np.float64)
    CHOLESKY_COV = np.linalg.cholesky(NOISE_COV)
    ANGULAR_VEL_NORM = 1/2.
    LINEAR_VEL_NORM = 1/2.
    POSITION_NORM = 1/2.

    POLICY_SHAPE = [18, 64, 64, 4]
    VALUE_SHAPE  = [18, 64, 64, 1]


class Utils:
    @staticmethod
    def noise(num):
        return np.array((np.random.normal(0, 1, size=(num, 4))) * np.float64(Config.CHOLESKY_COV))

    @staticmethod
    def normalize_states(states):
        n_state = np.array(states)
        n_state[:, 9:12]  = n_state[:, 9:12] * Config.POSITION_NORM
        n_state[:, 12:15] = n_state[:, 12:15] * Config.ANGULAR_VEL_NORM
        n_state[:, 15:18] = n_state[:, 15:18] * Config.LINEAR_VEL_NORM
        return n_state
    
    @staticmethod
    def forward(sess, network, states):
        prediction = network.model()
        states = Utils.normalize_states(states)
    
        return sess.run(prediction, feed_dict={network.input: states})

    @staticmethod
    def value_function_vectorized(costs, terminal_value):
        values = np.zeros(len(costs))
        values[-1] = terminal_value
    
        for i in range(len(costs)-2, -1, -1):
            values[i] = costs[i] + Config.DISCOUNT_VALUE * values[i+1]
    
        return values

    # Cost function as defined in formula 9
    @staticmethod
    def compute_cost(states, actions):
        states = np.array(states)
        actions = np.array(actions)
        position = 4. * 10**(-3) * np.sqrt(np.linalg.norm(states[:, 9:12], axis=1))
        angular = 5. * 10**(-5) * np.linalg.norm(states[:, 12:15], axis=1)
        linear = 5. * 10**(-5) * np.linalg.norm(states[:, 15:18], axis=1)
        action = 5. * 10**(-5) * np.linalg.norm(actions, axis=1)

        if len(action) != len(states) or len(linear) != len(states):
            print("Incorrect cost computation.")

        return position + action + angular + linear

