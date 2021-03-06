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
#        @date    10.10.2018                            #
#########################################################
import numpy as np
from pyquaternion import Quaternion

from .interface import Interface
from PyQuadSim.build import pyquadsim


class RaiInterface(Interface):
    @staticmethod
    def init():
        pyquadsim.createSimulator()

    @staticmethod
    def release():
        pyquadsim.release()

    @staticmethod
    def update_stateless(pose, actions, pid, dt):
        pose = pyquadsim.update(pose.tolist(), actions.tolist(), dt, 0)
        pose = np.array(pose, dtype=np.float64)
        return pose
