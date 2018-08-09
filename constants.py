import numpy as np

TRAJECTORIES_N = 256
BRANCHES_N = 128
INITIAL_LENGTH = 512
BRANCH_LENGTH = int(512)
NOISE_DEPTH = 2
VALUE_ITERATIONS = 30
VALUE_LOSS_LIMIT = 0.0001
DISCOUNT_VALUE = 0.99
TIME_STEP = 0.02
ACTION_BIAS = 18353
ACTION_MAX = 30000
NOISE_COV = np.matrix([ [5., 0., 0., 0.],
                        [0., 5., 0., 0.],
                        [0., 0., 5., 0.],
                        [0., 0., 0., 5.]], dtype=np.float32)
NOISE_STDDEV = 0.02
