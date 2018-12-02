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

from .interface import Interface
from PyQuadSim.build import pyquadsim

class IcarusInterface(Interface):
    @staticmethod
    def init():
        pyquadsim.createSimulator()

    @staticmethod
    def _get_pid():
        return pyquadsim.getPIDState()

    @staticmethod
    def _set_pid(pid):
        pyquadsim.setPIDState(pid)

    @staticmethod
    def release():
        pyquadsim.release()
        IcarusInterface.countSims = 0

    @staticmethod
    def update_stateless(pose, actions, pid, dt):
        IcarusInterface._set_pid(pid)
        pose = pyquadsim.update(pose.tolist(), actions.tolist(), dt)
        pose = np.array(pose, dtype=np.float64)
        return pose
