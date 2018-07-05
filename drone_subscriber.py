import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.msg import LinkState

class DroneSubscriber:
    """ Drone Subscriber to get data from Gazebo via ROS
    """
    def __init__(self):
        self.linear_velocity = (0,0,0)
        self.angular_velocity = (0,0,0)
        self.position = (0,0,0)
        #self.imu_sub = rospy.Subscriber('/hummingbird/imu', Imu, self.on_imu_topic)
        #self.pos_sub = rospy.Subscriber('/hummingbird/ground_truth/position', \
                                        #PointStamped, self.on_pos_topic)
        self.link = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

    def get_state(self):
        state = self.link(link_name='hummingbird/base_link', reference_frame='world').link_state
        linear = state.twist.linear
        self.linear_velocity = (linear.x, linear.y, linear.z)
        angular = state.twist.angular
        self.angular_velocity = (angular.x, angular.y, angular.z)
        position = state.pose.position
        self.position = (position.x, position.y, position.z)
        orientation = state.pose.orientation
        self.orientation = (orientation.w, orientation.x, orientation.y, orientation.z)

        return self.orientation, self.position, self.angular_velocity, self.linear_velocity

    def get_pose(self):
        base = self.link(link_name='hummingbird/base_link', reference_frame='world').link_state
        r0 = self.link(link_name='hummingbird/rotor_0', reference_frame='world').link_state
        r1 = self.link(link_name='hummingbird/rotor_1', reference_frame='world').link_state
        r2 = self.link(link_name='hummingbird/rotor_2', reference_frame='world').link_state
        r3 = self.link(link_name='hummingbird/rotor_3', reference_frame='world').link_state
        return (base, r0, r1, r2, r3)

    def on_imu_topic(self, msg):
        # save accelerometers
        lax = msg.linear_acceleration.x
        lay = msg.linear_acceleration.y
        laz = msg.linear_acceleration.z

        avx = msg.angular_velocity.x
        avy = msg.angular_velocity.y
        avz = msg.angular_velocity.z

        ox = msg.orientation.x
        oy = msg.orientation.y
        oz = msg.orientation.z
        ow = msg.orientation.w

        self.linear_acceleration = (lax, lay, laz)
        self.angular_velocity = (avx, avy, avz)
        self.orientation = (ow, ox, oy, oz)


    def on_pos_topic(self, msg):
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z

        self.position = (x, y, z)

