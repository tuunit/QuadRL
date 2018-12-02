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
import numpy as np
from pyquaternion import Quaternion

class Trajectory:
    _orientation = [i < 4 for i in range(13)]
    _position = [i >= 4 and i < 7 for i in range(13)]
    _angular_velocity = [(i >= 7 and i < 10) for i in range(13)]
    _linear_velocity = [i >= 10 for i in range(13)]

    def __init__(self, interface):
        self._interface = interface
        self._pose = self.random_pose()

    def step(self, actions, **kwargs):
        pose = self._interface.update_stateless(self._pose, actions, **kwargs)
        self._pose = np.array(pose, dtype=np.float64)
        return self._pose

    def snapshot(self):
        copy = Trajectory(self._interface)
        copy.set_pose(self._pose)
        return copy

    def reset(self):
        self._pose = np.array([0.0 for _ in range(13)])
        self._pose[self._interface.orientation] = [1.0, 0.0, 0.0, 0.0]
        self._pose = np.float64(self._pose)

    def set_pose(self, pose):
        if len(pose) != 13:
            raise TypeError('pose is required to contain 13 elements')
        self._pose = np.array(pose)

    def get_pose(self):
        return self._pose

    def get_state(self):
        orientation = self._pose[self._orientation]
        position = self._pose[self._position]
        angular = self._pose[self._angular_velocity]
        linear = self._pose[self._linear_velocity]

        orientation = np.ndarray.flatten(Quaternion(orientation).rotation_matrix)

        return np.concatenate((orientation, position, angular, linear))

    @staticmethod
    def random_pose():
        pose = np.array([0.0 for _ in range(13)])

        orientation = np.random.normal(0, 1, 4)
        orientation = orientation / np.linalg.norm(orientation)
        orientation[0] = np.abs(orientation[0])

        pose[Trajectory._position] = np.random.normal(0, 2, 3)
        pose[Trajectory._orientation] = orientation
        pose[Trajectory._linear_velocity] = np.random.normal(0, 2, 3)
        pose[Trajectory._angular_velocity] = np.random.normal(0, 2, 3)

        return np.float64(pose)
