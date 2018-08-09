import qsim
from time import sleep
from pyquaternion import Quaternion
import numpy as np


class DroneInterface:
    """ Drone Publisher to control Gazebo via ROS and send data to the simulator
    """

    def __init__(self):
        self.orientation = [i < 4 for i in range(13)]
        self.position = [i >= 4 and i < 7 for i in range(13)]
        self.angular_velocity = [(i >= 7 and i < 10) for i in range(13)]
        self.linear_velocity = [i >= 10 for i in range(13)]

        self.pose = np.array([0.0 for _ in range(13)])

        self.dt = float()
        qsim.init()

    def reset(self):
        self.pose = np.array([0.0 for _ in range(13)])
        self.pose[self.orientation] = [1.0, 0.0, 0.0, 0.0]
        self.pose[self.position][2] = 0.1

    def set_timestep(self, dt):
        self.dt = dt

    def set_pose(self, pose):
        self.pose = np.array(pose)

    def get_pose(self):
        return self.pose.tolist()

    def get_state(self):
        return tuple(self.pose[self.orientation].tolist()), tuple(self.pose[self.position].tolist()), \
               tuple(self.pose[self.angular_velocity].tolist()), tuple(self.pose[self.linear_velocity].tolist())

    def update(self, thrusts):
        thrusts = [x.item() for x in thrusts]
        pose = qsim.update(self.pose.tolist(), thrusts, self.dt)
        if pose is None:
            print(pose)
        else:
            self.pose = np.array(pose)
        #print(self.pose[self.position])
        #sleep(0.005)
        return self.get_pose()

    def initial_pose(self):
        self.reset()
        self.pose[self.position] = [np.random.normal(scale=3), np.random.normal(scale=3), np.random.rand() * 5 + 3]
        #self.pose[self.position] = [0, 0, 5]

        x = np.random.normal(scale=20)
        y = np.random.normal(scale=20)
        z = np.random.normal(scale=20)
        x = Quaternion(axis=[1, 0, 0], degrees=x)
        y = Quaternion(axis=[0, 1, 0], degrees=y)
        z = Quaternion(axis=[0, 0, 1], degrees=z)
        orientation = x * y * z
        self.pose[self.orientation] = orientation.elements

    def __del__(self):
        qsim.release()