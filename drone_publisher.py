import rospy
import roslaunch
from std_srvs.srv import Empty
from mav_msgs.msg import Actuators
from gazebo_msgs.srv import (DeleteModel, GetPhysicsProperties, SetLinkState, SetModelState)
from gazebo_msgs.msg import (ModelState, LinkState)
from time import sleep
from pyquaternion import Quaternion
import numpy as np

class DronePublisher:
    """ Drone Publisher to control Gazebo via ROS and send data to the simulator
    """
    def __init__(self):
        self.motor_pub = rospy.Publisher('/hummingbird/command/motor_speed', \
                                         Actuators, queue_size=10.0)
        self.set_link = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        self.set_model = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

    def reset(self):
        self.pause()
        sleep(0.01)
        self.reset_world()
        link = LinkState()
        link.pose.orientation.x = 0.0
        link.pose.orientation.y = 0.0
        link.pose.orientation.z = 0.0
        link.pose.orientation.w = 1.0
        link.pose.position.z = 0.01
        link.reference_frame = 'hummingbird/base_link'

        link.link_name = 'hummingbird/rotor_0'
        link.pose.position.x = 0.17
        link.pose.position.y = 0.0
        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(link)

        link.link_name = 'hummingbird/rotor_1'
        link.pose.position.x = 0.0
        link.pose.position.y = 0.17
        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(link)

        link.link_name = 'hummingbird/rotor_2'
        link.pose.position.x = -0.17
        link.pose.position.y = 0.0
        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(link)

        link.link_name = 'hummingbird/rotor_3'
        link.pose.position.x = 0.0
        link.pose.position.y = -0.17
        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(link)

    def set_pose(self, pose):
        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(pose[0])

        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(pose[1])

        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(pose[2])

        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(pose[3])

        rospy.wait_for_service('/gazebo/set_link_state')
        self.set_link(pose[4])

    def speed(self, velocities=[0,0,0,0]):
        if not isinstance(velocities, list):
            raise TypeError('Input must be a list')

        self.motor_pub.publish(angular_velocities=velocities)

    def initial_pose(self):
        model_state = ModelState()
        model_state.model_name = 'hummingbird'
        model_state.reference_frame = 'world'
        model_state.pose.position.z = np.random.rand()*5+3
        model_state.pose.position.x = np.random.normal(scale=3)
        model_state.pose.position.y = np.random.normal(scale=3)

        x = np.random.normal(scale=20)
        y = np.random.normal(scale=20)
        z = np.random.normal(scale=20)
        x = Quaternion(axis=[1, 0, 0], degrees=x)
        y = Quaternion(axis=[0, 1, 0], degrees=y)
        z = Quaternion(axis=[0, 0, 1], degrees=z)
        
        orientation = x * y * z
        orientation = orientation.elements

        model_state.pose.orientation.w = orientation[0]
        model_state.pose.orientation.x = orientation[1]
        model_state.pose.orientation.y = orientation[2]
        model_state.pose.orientation.z = orientation[3]
        model_state.twist.linear.z = np.random.rand()*10

        self.set_model(model_state)
