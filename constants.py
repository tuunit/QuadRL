import numpy as np

TRAJECTORIES_N = 256
BRANCHES_N = 64
INITIAL_LENGTH = BRANCHES_N * 16
BRANCH_LENGTH = int(BRANCHES_N * 8)
NOISE_DEPTH = 2
VALUE_ITERATIONS = 20
VALUE_LOSS_LIMIT = 0.0001
DISCOUNT_VALUE = 0.99
TIME_STEP = 0.02
ACTION_BIAS = 19000
ACTION_MAX = 30000
NOISE_COV = np.matrix([[0.8, 0.1, 0.1, 0.1],
                        [0.1, 0.8, 0.1, 0.1],
                        [0.1, 0.1, 0.8, 0.1],
                        [0.1, 0.1, 0.1, 0.8]], dtype=np.float32)
NOISE_STDDEV = 0.2
