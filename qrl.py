import os
import rospy
import pickle
import numpy as np
import tensorflow as tf
from PID import PID
from tqdm import tqdm
from random import sample
from time import (sleep, time)
from pyquaternion import Quaternion

from constants import *
from utils import *
from drone_publisher import DronePublisher
from drone_subscriber import DroneSubscriber
from neural_network import NeuralNet
from policy_network import PolicyNet
from value_network import ValueNet

# Main function to start the testing or training loop
def run(arguments):
    if arguments.test:
        run_test(arguments)
    else:
        run_training(arguments)

def run_training(arguments):
    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Initialize ROS node for communication with Gazebo
    rospy.init_node('training')

    # Instantiate publisher and subscriber for Gazebo
    pub = DronePublisher()
    sub = DroneSubscriber()

    # Instantiate PID controllers for nominal orientation
    pitch_PID = PID(30, 0, 15)
    roll_PID = PID(30, 0, 15)
    yaw_PID = PID(7.5, 0, 3.75)

    # Instantiate policy network with own Tensorflow graph
    policy_graph = tf.Graph()
    policy_sess = tf.Session(graph=policy_graph)

    policy_net = PolicyNet(shape=[18,64,64,4], graph=policy_graph)
    with policy_sess.as_default():
        with policy_sess.graph.as_default():
            tf.global_variables_initializer().run()
            policy_net.saver = tf.train.Saver()

    # Instantiate value network with own Tensorflow graph
    value_graph = tf.Graph()
    value_sess = tf.Session(graph=value_graph)

    value_net = ValueNet(shape=[18,64,64,1], graph=value_graph)
    with value_sess.as_default():
        with value_sess.graph.as_default():
            tf.global_variables_initializer().run()
            value_net.saver = tf.train.Saver()

    # List for recording trajectories
    history = []

    # Start rospy session
    while not rospy.is_shutdown():
        # Wait for rospy initialization
        sleep(0.1)

        # Start main training loop
        for t in range(0, TRAJECTORIES_N):

            if arguments.log:
                policy_log = open('policy_loss.txt', 'a')
                value_log = open('value_loss.txt', 'a')

            # List for the trajectories of the current training cycle
            trajectories = []

            print('Trajectory #' + str(t+1))

            # Reset simulation and PID controller
            pub.reset()
            pitch_PID.clear()
            roll_PID.clear()
            yaw_PID.clear()

            # Generate random target position for current training cycle
            TARGET = (np.random.randint(-10, 10), np.random.randint(-10, 10), np.random.randint(2,10))
            print('TARGET', TARGET)

            # Initialize random branch / junction points in time
            branches = sorted(sample(range(INITIAL_LENGTH), BRANCHES_N))
            branch_poses = []

            # Generate random start position for the Quadcopter
            pub.initial_pose()
            pub.unpause()

            # Temporary lists
            states = []
            actions = []
            costs = []

            # Initial flight
            for b in range(INITIAL_LENGTH):
                # Wait for simulation to happen
                sleep(0.05)

                # Get state information from drone subscriber
                orientation, position, angular, linear = sub.get_state()

                # Calculate relative distance to target position
                position = tuple(np.subtract(position, TARGET))

                # Calculate rotation matrix from quaternion
                orientation = Quaternion(orientation)
                [yaw, pitch, roll] = orientation.yaw_pitch_roll
                pitch_PID.update(pitch)
                roll_PID.update(roll)
                yaw_PID.update(yaw)

                orientation = np.ndarray.flatten(orientation.rotation_matrix)

                # Concatenate all tuples to generate an input state vector for the networks
                state = list(tuple(orientation) + position + angular + linear)

                # Predict action with policy network
                action = np.array(policy_net.model().eval(session=policy_sess, feed_dict={policy_net.input:[state]})[0])

                # Add PID controller outputs
                action[0] += -pitch_PID.output + yaw_PID.output
                action[1] += +roll_PID.output - yaw_PID.output
                action[2] += +pitch_PID.output + yaw_PID.output
                action[3] += -roll_PID.output - yaw_PID.output

                # Save prediction for the optimization step
                actions.append(action)

                # Add bias to guarantee take off
                action = action + 459

                # Clip output to guarantee a realistic simulation
                action = np.clip(action, 0, 900)

                # Save full pose of the Quadcopter if a junction has to be created at this point of time later on
                if b in branches:
                    branch_poses.append({'pose': sub.get_pose(), 'action': action, 'position': b})

                # Feed action vector to the drone
                pub.speed(list(action))

                # Save state vector
                states.append(state)

                # Calculate and save cost of state
                costs.append(cost(position, action, angular, linear))

            # Save initial trajectory to the list of the collection cycle
            trajectories.append({'level': -1, 'states': states, 'actions': actions, 'costs': costs})


            # Generate branch trajectories
            pbar = tqdm(range(len(branch_poses)*2))
            pbar.set_description('Generating branch trajectories')

            for i, b in enumerate(branch_poses):
                for n in range(0, NOISE_DEPTH):
                    pbar.update(1)

                    # Pause simulation and set starting pose of branch
                    pub.pause()
                    pub.set_pose(b['pose'])
                    sleep(0.05)
                    pub.unpause()

                    noise = None
                    states = []
                    actions = []
                    costs = []

                    # Start branch trajectory
                    for j in range(0, BRANCH_LENGTH):
                        # Wait for simulation to happen
                        sleep(0.05)

                        # Get state information from drone subscriber
                        orientation, position, angular, linear = sub.get_state()

                        # Calculate relative distance to target position
                        position = tuple(np.subtract(position, TARGET))

                        # Calculate rotation matrix from quaternion
                        orientation = Quaternion(orientation)
                        [yaw, pitch, roll] = orientation.yaw_pitch_roll
                        pitch_PID.update(pitch)
                        roll_PID.update(roll)
                        yaw_PID.update(yaw)

                        orientation = np.ndarray.flatten(orientation.rotation_matrix)

                        # Concatenate all tuples to generate an input state vector for the networks
                        state = list(tuple(orientation) + position + angular + linear)

                        # Predict action with policy network
                        action = np.array(policy_net.model().eval(session=policy_sess, feed_dict={policy_net.input:[state]})[0])

                        # Add PID controller outputs
                        action[0] += -pitch_PID.output + yaw_PID.output
                        action[1] += +roll_PID.output - yaw_PID.output
                        action[2] += +pitch_PID.output + yaw_PID.output
                        action[3] += -roll_PID.output - yaw_PID.output

                        # Save prediction for the optimization step
                        actions.append(action)

                        # Add bias to guarantee take off
                        action = action + 459

                        # Clip output to guarantee a realistic simulation
                        action = np.clip(action, 0, 900)

                        if n == j:
                            # Generate noise if first state of branch
                            noise = NOISE()
                            pub.speed(list(noise))
                        elif j < n:
                            # Repeat noise for previous junction trajectory states
                            pub.speed(list(trajectories[j-n]['noise']))
                        else:
                            # Feed action vector to the Quadcopter
                            pub.speed(list(action))

                            # Save state vector
                            states.append(state)

                            # Calculate and save cost of state
                            costs.append(cost(position, action, angular, linear))

                    # Save branch trajectory to the list of the collection cycle
                    trajectories.append({'level': n, 'noise': noise, 'states': states, 'actions': actions, 'costs': costs, 'position': b['position']})
            pbar.close()
            del(pbar)

            # Pause simulation while the training procedure
            pub.pause()

            # Get random trajectories from the previous collection cycles
            #random_traj = get_random_trajectories(history)

            #if random_traj:
            #    value_traj = trajectories + random_traj
            #else:
            value_traj = trajectories


            # Start value network training
            print('Value network training')
            pbar = tqdm(range(len(value_traj)))
            pbar.set_description('Calculating values')

            # Calculate desired values for every state of the recorded trajectories
            for i, trajectory in enumerate(value_traj):
                pbar.update(1)
                values = []
                for j in range(len(trajectory['states'])):
                    v = value_function(value_sess, value_net, trajectory['costs'], trajectory['states'], j)
                    values.append(v)
                trajectory['values'] = values
            pbar.close()
            del(pbar)

            # Optimize value network
            pbar = tqdm(range(VALUE_ITERATIONS))
            pbar.set_description('Optimizing value network')
            with value_sess.as_default():
                with value_sess.graph.as_default():
                    for i in range(VALUE_ITERATIONS):
                        pbar.update(1)
                        loss = train_value_network(value_sess, value_net, value_traj)
                        if arguments.log:
                            value_log.write(str(loss)+'\n')
                        if not i % 100:
                            print('Value Loss:', loss)
                        if loss < VALUE_LOSS_LIMIT:
                            break;
            print('Value Loss:', loss)
            pbar.close()
            del(pbar)

            # Optimize policy network
            print('Policy network training')
            with policy_sess.as_default():
                with policy_graph.as_default():
                    traj_len = len(trajectories)
                    #if random_traj:
                    #    policy_traj = trajectories + random_traj
                    #else:
                    policy_traj = trajectories
                    
                    pbar = tqdm(range(len(policy_traj)))
                    pbar.set_description('Optimising policy network')

                    param_update = 0
                    As = []
                    for i, trajectory in enumerate(policy_traj):
                        pbar.update(1)
                        if(trajectory['level'] >= 0):

                            # Get states of branch (off policy states)
                            states_f = trajectory['states']

                            # Get previous flight states (on policy states)
                            if trajectory['level'] == 1 and i < traj_len:
                                states_p = policy_traj[0]['states']
                            else:
                                states_p = policy_traj[i-1]['states']

                                if i >= traj_len and (i - traj_len) % 2 == 0:
                                    continue

                            if len(states_p) + policy_traj[i-1]['level'] != len(states_f) + trajectory['level']:
                                states_p = states_p[trajectory['position']:]

                            T_f = len(states_f)
                            T_p = len(states_p)
                            
                            # Predict value / reward of both trajectories termination state
                            with value_graph.as_default():
                                val_f = value_net.model().eval(session=value_sess, feed_dict={value_net.input:[states_f[T_f-1]]})[0][0]
                                val_p = value_net.model().eval(session=value_sess, feed_dict={value_net.input:[states_p[T_p-1]]})[0][0]

                            states = np.concatenate((states_f, states_p))
                            junction = states_p[0]
                            noise = trajectory['noise'] - 459
                            
                            # Feed the data to the optimization graph
                            A, nk = policy_sess.run([policy_net.A, policy_net.train_op], \
                                    feed_dict={policy_net.input: states, \
                                    policy_net.states_f_l: T_f, \
                                    policy_net.states_p_l: T_p, \
                                    policy_net.junction_state: junction, \
                                    policy_net.action_noise: noise, \
                                    policy_net.value_f: val_f, \
                                    policy_net.value_p: val_p})

                            # Save squared loss
                            As.append(A**2)

                            # Add up nk 
                            param_update += nk[0]

                    pbar.close()
                    del(pbar)
                    
                    # Apply learning rate to new update step
                    param_update = (0.01/(len(policy_traj) - 1)) * np.array(param_update)
                    print('param_update:', param_update)

                    if arguments.log:
                        policy_log.write(str(sum(As))+'\n')
                    print('Policy Loss:', sum(As))

                    
                    # Split update vector into parts and apply them to the Tensorflow variables
                    start = 0
                    grads_and_vars = []
                    for v in tf.trainable_variables():
                        if len(v.shape) == 2:
                            limit = v.shape[0].value * v.shape[1].value
                        else:
                            limit = v.shape[0].value
                        grads_and_vars.append((np.reshape(param_update[start:start+limit], v.shape), v))
                        start = start + limit
                    policy_net.optimizer.apply_gradients(grads_and_vars)

            # Save current collection cycle to history
            history.append(trajectories)

            # Save checkpoints of the current networks
            if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')

            timestamp = str(int(time()))
            with policy_sess.as_default():
                with policy_sess.graph.as_default():
                    policy_net.saver.save(policy_sess, 'checkpoints/policy_checkpoint_' + timestamp + '.ckpt')

            with value_sess.as_default():
                with value_sess.graph.as_default():
                    value_net.saver.save(value_sess, 'checkpoints/value_checkpoint_' + timestamp + '.ckpt')

def run_test(arguments):
    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Initialize ROS node for communication with Gazebo
    rospy.init_node('training')

    # Instantiate publisher and subscriber for Gazebo
    pub = DronePublisher()
    sub = DroneSubscriber()

    # Initialize policy network
    policy_net = PolicyNet(shape=[18,64,64,4])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, arguments.test)

        for _ in range(10):
            pub.pause()
            TARGET = (np.random.randint(-10, 10), np.random.randint(-10, 10), np.random.randint(2,10))
            print('TARGET', TARGET)

            # Generate random start position for the Quadcopter
            pub.initial_pose()
            pub.unpause()
            for _ in range(200):
                sleep(0.05)
                # Get state information from drone subscriber
                orientation, position, angular, linear = sub.get_state()

                # Calculate relative distance to target position
                position = tuple(np.subtract(position, TARGET))

                # Calculate rotation matrix from quaternion
                orientation = Quaternion(orientation)
                orientation = np.ndarray.flatten(orientation.rotation_matrix)

                # Concatenate all tuples to generate an input state vector for the networks
                state = list(tuple(orientation) + position + angular + linear)

                # Predict action with policy network
                action = np.array(sess.run(policy_net.model(), feed_dict={policy_net.input:[state]})[0])

                # Add bias to guarantee take off
                action = action + 459

                # Clip output to guarantee a realistic simulation
                action = np.clip(action, 0, 900)

                # Feed action vector to the drone
                pub.speed(list(action))


# Value function as defined in formula 4
def value_function(sess, value_net, costs, states, i):
    values = []
    T = len(costs)
    value_factor = value_net.model().eval(session=sess, feed_dict={value_net.input:[states[T-1]]})[0][0]
    for t in range(i, T):
        v = (DISCOUNT_VALUE**(t-i) * costs[t])
        values.append(v)
    return sum(values) + (DISCOUNT_VALUE**(T-i) * value_factor)

# Cost function as defined in formula 9
def cost(position, action, angular, linear):
    position = 4 * 10**(-3) * np.linalg.norm(position)
    action = 2 * 10**(-4) * np.linalg.norm(action)
    angular = 3 * 10**(-4) * np.linalg.norm(angular)
    linear = 5 * 10**(-4) * np.linalg.norm(linear)

    return position + action + angular + linear

def train_value_network(sess, value_net, trajectories):
    loss = 0
    for trajectory in trajectories:
        _, c = sess.run([value_net.train_op, value_net.loss], feed_dict={value_net.input: trajectory['states'], \
            value_net.output: trajectory['values']})
        loss += c
    return loss / len(trajectories)
