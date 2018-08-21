import numpy as np

TRAJECTORIES_N = 256
BRANCHES_N = 64
INITIAL_LENGTH = 1024
BRANCH_LENGTH = int(512)
NOISE_DEPTH = 1
VALUE_ITERATIONS = 200
VALUE_LOSS_LIMIT = 0.
DISCOUNT_VALUE = 0.99
TIME_STEP = 0.01
ACTION_BIAS = 18353
ACTION_MAX = 30000
ACTION_SCALE = 100.
NOISE_COV = np.matrix([[5., 0., 0., 0.],
                       [0., 5., 0., 0.],
                       [0., 0., 5., 0.],
                       [0., 0., 0., 5.]], dtype=np.float32)
CHOLESKY_COV = np.linalg.cholesky(NOISE_COV)
NOISE_STDDEV = 0.02
