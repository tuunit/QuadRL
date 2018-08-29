import numpy as np

TRAJECTORIES_N = 256
INITIAL_N = 512
BRANCHES_N = 1024
INITIAL_LENGTH = 512
BRANCH_LENGTH = int(256)
NOISE_DEPTH = 1
VALUE_ITERATIONS = 200
VALUE_LOSS_LIMIT = 0.0001
DISCOUNT_VALUE = 0.99
TIME_STEP = 0.01
ACTION_BIAS = 18353
ACTION_MAX = 30000
ACTION_SCALE = 60.
NOISE_COV = np.matrix([[6., 0., 0., 0.],
                       [0., 6., 0., 0.],
                       [0., 0., 6., 0.],
                       [0., 0., 0., 6.]], dtype=np.float32)
CHOLESKY_COV = np.linalg.cholesky(NOISE_COV)
NOISE_STDDEV = 0.02
ANGULAR_VEL_NORM = 6.
LINEAR_VEL_NORM = 5.
POSITION_NORM = 20.
