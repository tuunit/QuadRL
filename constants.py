import numpy as np

TRAJECTORIES_N = 512
INITIAL_N = 512
BRANCHES_N = 1024
INITIAL_LENGTH = 512
BRANCH_LENGTH = int(512)
TAIL_STEPS = 150
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
