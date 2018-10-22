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

from QuadNNLearn.build import pyquadsim


class IcarusInterface:
    """ Drone Publisher to control Gazebo via ROS and send data to the simulator
    """
    orientation = [i < 4 for i in range(13)]
    position = [i >= 4 and i < 7 for i in range(13)]
    angular_velocity = [(i >= 7 and i < 10) for i in range(13)]
    linear_velocity = [i >= 10 for i in range(13)]
    dt = float()

    @staticmethod
    def init():
        pyquadsim.createSimulator()

    @staticmethod
    def _get_pid(idx):
        return pyquadsim.getPIDState(idx)

    @staticmethod
    def _set_pid(idx, pid):
        pyquadsim.setPIDState(pid, idx)

    @staticmethod
    def set_timestep(dt):
        IcarusInterface.dt = dt

    @staticmethod
    def get_state(pose):
        return pose[IcarusInterface.orientation], pose[IcarusInterface.position], \
               pose[IcarusInterface.angular_velocity], pose[IcarusInterface.linear_velocity]

    @staticmethod
    def release():
        pyquadsim.release()
        IcarusInterface.countSims = 0

    @staticmethod
    def update_stateless(pose, pid, thrusts):
        IcarusInterface._set_pid(0, pid)
        pose = pyquadsim.update(pose.tolist(), thrusts.tolist(), IcarusInterface.dt, 0)
        pid = IcarusInterface._get_pid(0)
        if pose is not None:
            pose = np.array(pose, dtype=np.float64)
        return pose, pid

    @staticmethod
    def random_pose():
        pose = np.array([0.0 for _ in range(13)])

        orientation = np.random.normal(0, 1, 4)
        orientation = orientation / np.linalg.norm(orientation)
        orientation[0] = np.abs(orientation[0])

        pose[IcarusInterface.position] = np.random.normal(0, 2, 3)
        pose[IcarusInterface.orientation] = orientation
        pose[IcarusInterface.linear_velocity] = np.random.normal(0, 2, 3)
        pose[IcarusInterface.angular_velocity] = np.random.normal(0, 2, 3)

        return np.float64(pose)


    @staticmethod
    def get_pose_with_rotation_mat(pose):
        orientation, position, angular, linear = IcarusInterface.get_state(pose)
        orientation = np.ndarray.flatten(Quaternion(orientation).rotation_matrix.transpose())
        return np.concatenate((orientation, position, angular, linear))

