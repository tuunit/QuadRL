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
    TRAJECTORIES_N = 512
    INITIAL_N = 512
    BRANCHES_N = 1024
    INITIAL_LENGTH = 512
    BRANCH_LENGTH = int(512)
    TAIL_STEPS = 128
    NOISE_DEPTH = 1
    VALUE_ITERATIONS = 300
    VALUE_LOSS_LIMIT = 0.0001
    DISCOUNT_VALUE = 0.99
    TIME_STEP = 0.01
    ACTION_BIAS = .00
    ACTION_MAX = 30000
    ACTION_SCALE = 1.
    NOISE_COV = np.matrix([[.05, 0., 0., 0.],
                           [0., .05, 0., 0.],
                           [0., 0., .05, 0.],
                           [0., 0., 0., .05]], dtype=np.float64)
    CHOLESKY_COV = np.linalg.cholesky(NOISE_COV)
    ANGULAR_VEL_NORM = 1/2.
    LINEAR_VEL_NORM = 1/2.
    POSITION_NORM = 1/5.

def NOISE():
    return np.array((np.random.normal(0, 1, 4)) * np.float64(Config.CHOLESKY_COV))[0]

def NOISE_MAT(num):
    return np.array((np.random.normal(0, 1, size=(num, 4))) * np.float64(Config.CHOLESKY_COV))
