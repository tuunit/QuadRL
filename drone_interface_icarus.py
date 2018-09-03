from time import sleep
from pyquaternion import Quaternion
import numpy as np
from queue import Queue
from QuadSimulator import stepSim

class DroneInterface:
    """ Drone Publisher to control Gazebo via ROS and send data to the simulator
    """
    orientation = [i < 4 for i in range(13)]
    position = [i >= 4 and i < 7 for i in range(13)]
    angular_velocity = [(i >= 7 and i < 10) for i in range(13)]
    linear_velocity = [i >= 10 for i in range(13)]
    dt = float()

    @staticmethod
    def init():
        qsim.createSimulator()

    @staticmethod
    def _get_pid(idx):
        return qsim.getPIDState(idx)

    @staticmethod
    def _set_pid(idx, pid):
        qsim.setPIDState(pid, idx)

    @staticmethod
    def set_timestep(dt):
        DroneInterface.dt = dt

    @staticmethod
    def get_state(pose):
        return pose[DroneInterface.orientation], pose[DroneInterface.position], \
               pose[DroneInterface.angular_velocity], pose[DroneInterface.linear_velocity]

    @staticmethod
    def release():
        qsim.release()
        DroneInterface.countSims = 0

    @staticmethod
    def update_stateless(pose, pid, thrusts):
        #DroneInterface._set_pid(0, pid)
        pose = stepSim(pose, thrusts)
        #qsim.update(pose.tolist(), thrusts.tolist(), DroneInterface.dt, 0)
        #pid = DroneInterface._get_pid(0)
        if pose is not None:
            pose = np.array(pose, dtype=np.float64)
        return pose, pid

    @staticmethod
    def random_pose():
        pose = np.array([0.0 for _ in range(13)])
        pose[DroneInterface.orientation] = [1.0, 0.0, 0.0, 0.0]

        pose[DroneInterface.position] = (np.random.rand(3) * 10) - 5
        #pose[DroneInterface.position] = [0, 0, 5]

        x, y, z = (np.random.rand(3) * 40) - 20
        x = Quaternion(axis=[1, 0, 0], degrees=x)
        y = Quaternion(axis=[0, 1, 0], degrees=y)
        z = Quaternion(axis=[0, 0, 1], degrees=z)
        orientation = x * y * z
        pose[DroneInterface.orientation] = orientation.elements
        pose[DroneInterface.linear_velocity] = (np.random.rand(3) * 20) - 10
        pose[DroneInterface.angular_velocity] = (np.random.rand(3) * 6) - 3
        return np.float64(pose)

    @staticmethod
    def get_pose_with_rotation_mat(pose):
        orientation, position, angular, linear = DroneInterface.get_state(pose)
        orientation = np.ndarray.flatten(Quaternion(orientation).rotation_matrix.transpose())
        return np.concatenate((orientation, position, angular, linear))


class Trajectory:
    def __init__(self):
        self.pose = DroneInterface.random_pose()
        self.pid = [0.0 for _ in range(12)]

    def reset(self):
        self.pose = np.array([0.0 for _ in range(13)])
        self.pose[DroneInterface.orientation] = [1.0, 0.0, 0.0, 0.0]
        self.pose[DroneInterface.position][2] = 0.1
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
        return DroneInterface.get_state(self.pose)

    def get_pose_with_rotation_mat(self):
        return DroneInterface.get_pose_with_rotation_mat(self.pose)

    def set_pose_with_rotation_mat(self, pose):
        rot_mat = np.array([pose[:3], pose[3:6], pose[6:9]]).transpose()
        u, s, vt = np.linalg.svd(rot_mat)
        rot_mat = np.matmul(u, np.matmul(np.eye(3), vt))
        self.pose = np.concatenate((np.array(Quaternion(matrix=rot_mat).elements), pose[9:]))

    def step(self, thrusts):
        pose, pid = DroneInterface.update_stateless(self.pose, self.pid, thrusts)
        if pose is None:
            print(pose)
        else:
            self.pose = np.array(pose, dtype=np.float64)
            self.pid = pid
        return self.get_pose

    def snapshot(self):
        copy = Trajectory()
        copy.set_pid(self.get_pid())
        copy.set_pose(self.get_pose())
        return copy

