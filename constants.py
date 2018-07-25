import numpy as np

TRAJECTORIES_N = 256
BRANCHES_N = 128
INITIAL_LENGTH = BRANCHES_N * 2
BRANCH_LENGTH = BRANCHES_N / 2
NOISE_DEPTH = 2
VALUE_ITERATIONS = 200
VALUE_LOSS_LIMIT = 0.0001
DISCOUNT_VALUE = 0.99
NOISE_COV = np.array(np.diag([800,800,800,800]), dtype=np.float32)