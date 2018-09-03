import numpy as np
from pyquaternion import Quaternion
from constants import *

length_ = 0.17
dragCoeff_ = 0.016
mass_ = 0.665
gravity_ = np.array([0., 0., -9.81]).reshape([-1, 1])
inertia_ = np.diag(np.array([0.007, 0.007, 0.012]))
inertiaInv_ = np.linalg.inv(inertia_)

transsThrust2GenForce = np.mat([
    [0,             0,          length_,        -length_    ],
    [-length_,      length_,    0,              0           ],
    [dragCoeff_,    dragCoeff_, -dragCoeff_,    -dragCoeff_ ],
    [1,             1,          1,              1           ]])

transsThrust2GenForceInv = np.linalg.inv(transsThrust2GenForce)

def skewM(vec):
    return np.mat([
        [0.0, -vec[2], vec[1]],
        [vec[2], 0.0, -vec[0]],
        [-vec[1], vec[0], 0.0]])

def expM(vec):
    angle = np.linalg.norm(vec)
    if angle < 1e-10:
        return np.identity(3)
    axis = vec / angle
    vecSkew = skewM(axis)
    return np.identity(3) + np.sin(angle) * vecSkew + (1.0 - np.cos(angle)) * vecSkew * vecSkew

def boxplusI_Frame(quat, rotation):
    rotmat = expM(rotation)
    quat2 = Quaternion(matrix=rotmat)
    return np.array((quat2 * quat).normalised.elements)

def stepSim(state, action):
    if len(state) != 13 or len(action) != 4:
        print("Invalid arguments passed to QuadSimulator::stepSim!")
        exit(0)

    state = np.array(state).reshape([-1, 1])
    action = np.array(action).reshape([-1, 1])

    orientation = state[:4]
    angle = 2.0 * np.arccos(orientation[0][0])
    position = state[4:7]
    rotation = np.mat(Quaternion(orientation).rotation_matrix)
    velocities = state[7:13]
    w_B_ = rotation.transpose() * velocities[:3]

    kp_rot = -0.2
    kd_rot = -0.06

    if angle > 1e-6:
        fbTorque_b = kp_rot * angle * (rotation.transpose() * state[1:4]) / np.sin(angle) + kd_rot * (rotation.transpose() * velocities[:3])
    else:
        fbTorque_b = kd_rot * (rotation.transpose() * velocities[:3])
    fbTorque_b[2] = fbTorque_b[2] * 0.15

    actionGenForce = transsThrust2GenForce * ACTION_SCALE * action
    B_torque = actionGenForce[:3] + fbTorque_b
    B_force = np.array([0., 0., actionGenForce[3]]).reshape([-1, 1])
    B_force[2][0] += mass_ * 9.81

    genForce = np.concatenate((B_torque, B_force[2:3]))
    thrust = transsThrust2GenForceInv * genForce
    thrust = np.maximum(1e-8, thrust)
    genForce = transsThrust2GenForce * thrust

    B_torque = genForce[:3]
    B_force[2] = genForce[3]

    d_vel = np.zeros(6).reshape([-1, 1])
    d_vel[-3:] = (rotation * B_force) / mass_ + gravity_
    d_vel[:3] = rotation * (inertiaInv_ * (B_torque - np.cross(w_B_.reshape([1, -1]), (inertia_ * w_B_).reshape([1, -1])).reshape([-1, 1])))

    velocities += d_vel * TIME_STEP

    w_IXdt_ = velocities[:3] * TIME_STEP
    orientation = boxplusI_Frame(Quaternion(orientation), w_IXdt_)

    position = position + velocities[-3:] * TIME_STEP

    velocities[:3] = np.maximum(np.minimum(velocities[:3], 20), -20)
    velocities[-3:] = np.maximum(np.minimum(velocities[-3:], 5), -5)

    return np.concatenate((orientation.reshape([-1]), position.reshape([-1]), velocities.reshape([-1])))
