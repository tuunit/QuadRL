#!/usr/bin/python
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
import argparse
import numpy as np

import qrl

parser = argparse.ArgumentParser(description='Quadcopter Reinforcment Learning')
parser.add_argument('--log', action='store_true', help='Log policy and value network losses')
parser.add_argument('--test', nargs='?', metavar='FILE PATH', default=False, const=True, type=str, help='Test policy network')
parser.add_argument('--restore', nargs='?', metavar='FILE PATH', default=False, const=True, type=str, help='Restore network parameters')

if __name__ == '__main__':
    qrl.run(parser.parse_args())
