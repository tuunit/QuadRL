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
import math
import numpy as np
from pyquaternion import Quaternion

from .interface import Interface

class PythonSimulator(Interface):
    @staticmethod
    def update_stateless(pose, actions):
        # config
        dt = 0.01
        g = 9.80665

        kp_rot = -0.08
        kd_rot = -0.0002

        mass = 34.1e-3
        length = 0.046
        inertia = np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5])

        kF = 8.93585e-7# [ N * s^2 ]
        kM = 5.161e-9# [ N * m * s^2 ]
        km = kM / kF # thrust to drag gain

        fMin = 0
        rpsMax = 25000 / 60.;
        fMax = kF * rpsMax * rpsMax # [ N * s^2 ] * [ s^-1 ] * [ s^-1 ] = [ N ] = [ (kg * m) / s^2 ]

        aMax = 4.363323129985824 # [ rad / s ]

        pose = np.reshape(pose, [-1, 1])
        actions = np.reshape(actions, [-1, 1])

        orientation = pose[:4]
        position = pose[4:7]
        angular_velocity = pose[7:10]
        linear_velocity = pose[10:13]

        # get rotation angle from quaternion for PD controller
        angle = 2.0 * math.acos(orientation[0][0])


        R = np.mat(Quaternion(orientation).rotation_matrix)

        # body rates
        brates = R.transpose() * angular_velocity

        # xmode
        l = math.sqrt(2.) / 2. * length;
        eqSystemMat = np.matrix([[ l,  -l,   l,  -l],
                                 [-l,   l,   l,  -l],
                                 [km,  km, -km, -km],
                                 [1.,  1.,  1.,  1.]])

        # +mode
        #double l = length;
        #eqSystemMat = np.matrix([[0.,   l,  0.,  -l],
        #                         [-l,  0.,   l,  0.],
        #                         [km, -km,  km, -km],
        #                         [1.,  1.,  1.,  1.]])

        if angle > 1e-6:
            torque_pd = kp_rot * angle * (R.transpose() * pose[1:4]) / math.sin(angle) + kd_rot * (R.transpose() * angular_velocity)
        else:
            torque_pd = kd_rot * (R.transpose() * angular_velocity)

        # Lower yaw gains
        torque_pd[2] *= 0.15

        # convert forces to roll, pitch, yaw and thrust
        forces = eqSystemMat * actions

        # add torque from PD controller to torque
        torque = forces[0:3] + torque_pd

        # add force of gravity
        force = np.array([0., 0., forces[3] + mass * g]).reshape([-1, 1])

        thrust =  np.linalg.inv(eqSystemMat) * np.concatenate((torque, force[2:3]))
        thrust = np.maximum(fMin, thrust)
        forces = eqSystemMat * thrust

        torque = forces[:3] 
        force[2] = forces[3]

        linear_acc = (R * force) / mass + np.reshape([0., 0., g], [-1, 1])
        angular_acc = R * (np.linalg.inv(inertia) * (torque - np.cross(brates.reshape([1, -1]), (inertia * brates).reshape([1, -1])).reshape([-1, 1])))

        # update pose after time step
        linear_velocity += linear_acc * dt
        angular_velocity += angular_acc * dt

        orientation = Quaternion(orientation)
        vec = angular_velocity * dt
        angle = np.linalg.norm(vec)
        if angle < 1e-10:
            rotmat = np.identity(3)
        else:
            axis = vec / angle
            vecSkew = np.mat([[0.0, -axis[2], axis[1]],
                              [axis[2], 0.0, -axis[0]],
                              [-axis[1], axis[0], 0.0]])
            rotmat = np.identity(3) + math.sin(angle) * vecSkew + (1.0 - math.cos(angle)) * vecSkew * vecSkew

        orientation = Quaternion(matrix=rotmat) * orientation
        orientation = np.reshape(orientation.normalised.elements, [-1])

        position += linear_velocity * dt
        position = np.reshape(position, [-1])
        angular_velocity = np.clip(angular_velocity, -aMax, aMax).reshape([-1])
        linear_velocity = np.reshape(linear_velocity, [-1])

        pose = np.concatenate((orientation, position, angular_velocity, linear_velocity))

        return pose


