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

from .interface import IcarusInterface

class Trajectory:
    def __init__(self):
        self.pose = IcarusInterface.random_pose()
        self.pid = [0.0 for _ in range(12)]

    def reset(self):
        self.pose = np.array([0.0 for _ in range(13)])
        self.pose[IcarusInterface.orientation] = [1.0, 0.0, 0.0, 0.0]
        self.pose = np.float64(self.pose)

    def set_pose(self, pose):
        self.pose = np.array(pose)

    def get_pose(self):
        return self.pose

    def set_pid(self, pid):
        self.pid = pid

    def get_pid(self):
        return self.pid

    def get_state(self):
        return IcarusInterface.get_state(self.pose)

    def get_pose_with_rotation_mat(self):
        return IcarusInterface.get_pose_with_rotation_mat(self.pose)

    def step(self, thrusts):
        pose, pid = IcarusInterface.update_stateless(self.pose, self.pid, thrusts)
        if pose is None:
            pass
        else:
            self.pose = np.array(pose, dtype=np.float64)
            self.pid = pid
        return self.get_pose

    def snapshot(self):
        copy = Trajectory()
        copy.set_pid(self.get_pid())
        copy.set_pose(self.get_pose())
        return copy

