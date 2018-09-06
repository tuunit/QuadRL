import numpy as np
from constants import *
from random import sample

def NOISE():
    return np.array((np.random.normal(0, 1, 4)) * np.float64(CHOLESKY_COV))[0]

def NOISE_MAT(num):
    return np.array((np.random.normal(0, 1, size=(num, 4))) * np.float64(CHOLESKY_COV))

def timer(x):
    hours, rem = divmod(x, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.4f}".format(int(hours),int(minutes),seconds)

def get_random_trajectories(trajectories):
    if not trajectories:
        return None

    traj_n = sum([len(x) for x in trajectories])
    samples = sorted(sample(range(traj_n), BRANCHES_N))

    traj_samples = []
    count = 0
    for i in range(len(trajectories)):
        for j in range(len(trajectories[i])):
            if count in samples:
                traj_samples.append(trajectories[i][j])

                if trajectories[i][j]['level'] == 1:
                    traj_samples.append(trajectories[i][0])
                elif trajectories[i][j]['level'] > 1:
                    traj_samples.append(trajectories[i][j-1])
            count = count + 1

    return traj_samples
