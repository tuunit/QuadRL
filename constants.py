import numpy as np

TRAJECTORIES_N = 256
INITIAL_N = 512
BRANCHES_N = 1024
INITIAL_LENGTH = 500
BRANCH_LENGTH = int(500)
TAIL_STEPS = 150
NOISE_DEPTH = 1
VALUE_ITERATIONS = 300
VALUE_LOSS_LIMIT = 0.0001
DISCOUNT_VALUE = 0.99
TIME_STEP = 0.01
ACTION_BIAS = .00
ACTION_MAX = 30000
ACTION_SCALE = 2.
NOISE_COV = np.matrix([[.2, 0., 0., 0.],
                       [0., .2, 0., 0.],
                       [0., 0., .2, 0.],
                       [0., 0., 0., .2]], dtype=np.float32)
CHOLESKY_COV = np.linalg.cholesky(NOISE_COV)
NOISE_STDDEV = 0.02
ANGULAR_VEL_NORM = 1.
LINEAR_VEL_NORM = 1.
POSITION_NORM = 1.
