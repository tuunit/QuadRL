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
#        @date    26.08.2018                            #
#########################################################
import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from random import sample
from time import (sleep, time)
from pyquaternion import Quaternion
from gui import GUI
from constants import *
from utils import *
from drone_interface_icarus import DroneInterface, Trajectory
from neural_network import NeuralNet
from policy_network import PolicyNet
from value_network import ValueNet
from queue import Queue
import multiprocessing

# Value function as defined in formula 4
def value_function(sess, value_net, costs, states, i):
    values = []
    T = len(costs)
    value_factor = value_net.model().eval(session=sess, feed_dict={value_net.input:[states[T-1]]})[0][0]
    for t in range(i, T-1):
        v = DISCOUNT_VALUE**(t-i) * costs[t]
        values.append(v)
    return sum(values) + (DISCOUNT_VALUE**(T-(i+1)) * value_factor)

def value_function_vectorized(costs, terminal_value):
    values = [terminal_value]
    next_value = terminal_value

    for i in range(len(costs)-2, -1, -1):
        next_value = costs[i] + DISCOUNT_VALUE*next_value
        values.append(next_value)

    return list(reversed(values))

# Cost function as defined in formula 9
def cost(position, action, angular, linear):
    position = 4. * 10**(-3) * np.linalg.norm(position)
    angular = 5. * 10**(-4) * np.linalg.norm(angular)
    linear = 5. * 10**(-4) * np.linalg.norm(linear)
    action = 1. * 10**(-4) * np.linalg.norm(action)

    return position + action + angular + linear

# Cost function as defined in formula 9
def compute_cost_mat(states, actions):
    position = 4. * 10**(-3) * np.linalg.norm(states[:, 9:12], axis=1)
    angular = 5. * 10**(-4) * np.linalg.norm(states[:, 12:15], axis=1)
    linear = 5. * 10**(-4) * np.linalg.norm(states[:, 15:18], axis=1)
    action = 1. * 10**(-4) * np.linalg.norm(actions, axis=1)

    return position + action + angular + linear

def normalize_state(state):
    n_state = np.array(state)
    n_state[9:12] = state[9:12] / POSITION_NORM
    n_state[12:15] = state[12:15] / ANGULAR_VEL_NORM
    n_state[15:18] = state[15:18] / LINEAR_VEL_NORM
    return n_state

def normalize_states_mat(states):
    n_state = np.array(states)
    n_state[:, 9:12] = states[:, 9:12] / POSITION_NORM
    n_state[:, 12:15] = states[:, 12:15] / ANGULAR_VEL_NORM
    n_state[:, 15:18] = states[:, 15:18] / LINEAR_VEL_NORM
    return n_state


def train_value_network(sess, value_net, trajectories):
    states = []
    values = []
    for traj in trajectories:
        states.append(normalize_state(traj['states'][0]))
        values.append(traj['values'][0])
    values = np.array(values).reshape([-1, 1])
    _, c = sess.run([value_net.train_op, value_net.loss], feed_dict={value_net.input: states,
                    value_net.output: values})
    return c

# Main function to start the testing or training loop
def run(arguments):
    if arguments.test:
        run_test(arguments)
    else:
        run_training(arguments)

def traj_step(args):
    traj, thrusts = args
    traj.step(thrusts)
    return traj

def visualize(trajectories):
    def state_to_dict(state):
        mat = [state[0:3], state[3:6], state[6:9]]
        return {'position': state[9:12], 'rotation_matrix': mat}

    gui = GUI(0.8)
    b = 1
    for i in range(len(trajectories[-1]['states']) + trajectories[-1]['position'] + trajectories[b]['level']):
        for j in range(b, len(trajectories)):
            if trajectories[b]['position'] + trajectories[b]['level'] + 1 == i:
                gui.add_quadrotor()
                b += 1
        for j in range(b):
            if j == 0:
                idx = i
            else:
                idx = i - (trajectories[j]['position'] + trajectories[j]['level'] + 1)
            if idx >= len(trajectories[j]['states']):
                continue

            gui.update(state_to_dict(trajectories[j]['states'][idx]), j)

def run_training(arguments):
    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Instantiate policy network with own Tensorflow graph
    policy_graph = tf.Graph()
    policy_sess = tf.Session(graph=policy_graph)

    policy_net = PolicyNet(shape=[18, 64, 64, 4], graph=policy_graph)
    with policy_sess.as_default():
        with policy_sess.graph.as_default():
            tf.global_variables_initializer().run()
            policy_net.saver = tf.train.Saver()
            #policy_net.saver.restore(policy_sess,
            #                         'checkpoints/policy_checkpoint_1534956695.ckpt')

    # Instantiate value network with own Tensorflow graph
    value_graph = tf.Graph()
    value_sess = tf.Session(graph=value_graph)

    value_net = ValueNet(shape=[18, 64, 64, 1], graph=value_graph)
    with value_sess.as_default():
        with value_sess.graph.as_default():
            tf.global_variables_initializer().run()
            value_net.saver = tf.train.Saver()
            #value_net.saver.restore(value_sess,
            #                        'checkpoints/value_checkpoint_1534956695.ckpt')

    if arguments.log:
        policy_log = open('policy_loss.txt', 'a')
        value_log = open('value_loss.txt', 'a')
        policy_log.write('----Start----\n')
        value_log.write('----Start----\n')

    DroneInterface.set_timestep(TIME_STEP)

    # Start main training loop
    for t in range(0, TRAJECTORIES_N):

        if arguments.log:
            policy_log = open('policy_loss.txt', 'a')
            value_log = open('value_loss.txt', 'a')

        # List for the trajectories of the current training cycle
        trajectories = []

        print('Trajectory #' + str(t+1))

        # Generate random target position for current training cycle
        TARGET = (np.random.randint(-10, 10), np.random.randint(-10, 10), np.random.randint(2, 10))
        TARGET = np.array([0, 0, 10], dtype=np.float64)

        print('TARGET', TARGET)

        # Initialize random branch / junction points in time
        branches = sorted(sample(range(INITIAL_LENGTH - BRANCH_LENGTH), BRANCHES_N))
        branch_trajs = {'trajs': [], 'positions': []}

        # Temporary lists
        states = []
        actions = []
        costs = []

        actions_sum = [0., 0., 0., 0.]
        actions_count = 0

        DroneInterface.init()
        initial_traj = Trajectory()
        initial_traj.set_pose(DroneInterface.random_pose())

        # Initial flight
        for b in range(INITIAL_LENGTH):
            # Get state information from drone subscriber
            orientation, position, angular, linear = initial_traj.get_state()

            # Calculate relative distance to target position
            position = np.subtract(position, TARGET)
            if b == 0:
                print(position)

            orientation = np.ndarray.flatten(Quaternion(orientation).rotation_matrix)

            # Concatenate all to generate an input state vector for the networks
            state = np.concatenate((orientation, position, angular, linear))

            # Predict action with policy network
            action = np.array(policy_net.model().eval(session=policy_sess, feed_dict={policy_net.input:[normalize_state(state)]})[0], dtype=np.float64)
            _action = action
            # Save prediction for the optimization step

            # Calculate and save cost of state
            costs.append(cost(position, action, angular, linear))

            # Add bias to guarantee take off
            action = ACTION_SCALE * action + ACTION_BIAS

            actions.append(action)
            actions_sum += np.absolute(_action)
            actions_count += 1

            # Save full pose of the Quadcopter if a junction has to be created at this point of time later on
            if b in branches:
                for _ in range(NOISE_DEPTH):
                    branch = initial_traj.snapshot()
                    branch_trajs['trajs'].append(branch)
                    branch_trajs['positions'].append(b)

            # Save state vector
            states.append(state)

            # Feed action vector to the drone
            initial_traj.step(action)

            if b == INITIAL_LENGTH -1:
                terminal_value = position
        # Save initial trajectory to the list of the collection cycle
        trajectories.append({'level': -1, 'states': states, 'actions': actions, 'costs': costs, 'order': -1})

        ################################################################################################################
        ###################################    PARALLEL SIMULATION: STARTS    ##########################################
        ################################################################################################################

        # Generate branch trajectories
        pbar = tqdm(range(BRANCH_LENGTH))
        pbar.set_description('Generating branch trajectories')

        states_mat = []
        actions_mat = []
        costs_mat = []
        noise_mat = []
        states = []
        for i, b in enumerate(branch_trajs['trajs']):
            states.append(b.get_pose_with_rotation_mat())

        states = np.array(states)

        with multiprocessing.Pool(processes=6) as pool:
            for j in range(0, BRANCH_LENGTH):
                pbar.update(1)

                actions = np.array(policy_net.model().eval(session=policy_sess, feed_dict={policy_net.input: normalize_states_mat(states)}), dtype=np.float64)

                states_mat.append(states)
                actions_mat.append(actions)

                if j < NOISE_DEPTH:
                    mask = np.array([False if (x % NOISE_DEPTH) >= j else True for x in range(BRANCHES_N * NOISE_DEPTH)])
                    noises = NOISE_MAT(BRANCHES_N)
                    noises = np.repeat(noises, NOISE_DEPTH, axis=0)
                    noises[mask] *= 0.
                    noises = noises + actions

                    actions_feed = ACTION_SCALE*noises + ACTION_BIAS
                    costs = compute_cost_mat(states, actions)
                    costs[~mask] *= 0.
                    costs_mat.append(costs)

                    mask = np.array([False if (x % NOISE_DEPTH) == j else True for x in range(BRANCHES_N * NOISE_DEPTH)])
                    noises[mask] *= 0.
                    noise_mat.append(noises)
                else:
                    actions_feed = ACTION_SCALE*actions + ACTION_BIAS
                    costs_mat.append(compute_cost_mat(states, actions))

                branch_trajs['trajs'] = pool.map(traj_step, zip(branch_trajs['trajs'], actions_feed))

                states = np.array([traj.get_pose_with_rotation_mat() for traj in branch_trajs['trajs']])

        noise_mat = np.array(noise_mat)
        states_mat = np.array(states_mat)
        actions_mat = np.array(actions_mat)
        costs_mat = np.array(costs_mat)

        for i, b in enumerate(branch_trajs['positions']):
            n = i % NOISE_DEPTH

            noise = noise_mat[n, i]
            actions = actions_mat[n + 1:, i]
            states = states_mat[n + 1:, i]
            costs = costs_mat[n + 1:, i]
            if len(actions) != len(states) or len(states) != len(costs) or len(states) != BRANCH_LENGTH - (n + 1):
                print('ERROR: Anomalous trajectory data.')

            trajectories.append({'level': n, 'noise': noise, 'states': states, 'actions': actions, 'costs': costs, 'position': b})

        pbar.close()
        del(pbar)

        #visualize(trajectories)
        value_traj = trajectories

        # Value calculations
        terminal_states = [normalize_state(trajectory['states'][-1]) for trajectory in value_traj]
        terminal_values = value_net.model().eval(session=value_sess, feed_dict={value_net.input: terminal_states})
        for i, trajectory in enumerate(value_traj):
            trajectory['values'] = value_function_vectorized(trajectory['costs'], terminal_values[i])
        del terminal_values
        del terminal_states


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
                    if not i % (VALUE_ITERATIONS/5):
                        pbar.write("Value loss: {:.4f}".format(loss))
                    pbar.set_postfix(loss="{:.4f}".format(loss))
                    if loss <= VALUE_LOSS_LIMIT:
                        break
        pbar.close()
        del(pbar)

        # Start value network training
        print('Value network training')

        print('Mean Action Vector:', actions_sum / actions_count)
        print('Terminal position for Initial Trajectory:', terminal_value, np.linalg.norm(terminal_value))
        print('Value for Initial Trajectory:', trajectories[0]['values'][0])
        print('Approximated Value for Initial Trajectory:', value_net.model().eval(session=value_sess, feed_dict={value_net.input: [normalize_state(trajectories[0]['states'][0])]})[0])

        # Advantages and their gradients w.r.t action computed here.
        As = []
        grad_As = []
        junction_states = []

        for i, trajectory in enumerate(trajectories):
            if trajectory['level'] >= 0:
                vf = trajectory['values'][0]
                noise = trajectory['noise']

                if trajectory['level'] == 0:
                    vp = trajectories[0]['values'][trajectory['position']]
                    junction_state = trajectories[0]['states'][trajectory['position']]
                    junction_action = trajectories[0]['actions'][trajectory['position']]
                else:
                    vp = trajectories[i - 1]['values'][0]
                    junction_state = trajectories[i - 1]['states'][0]
                    junction_action = trajectories[i - 1]['actions'][0]

                rf = cost(junction_state[9:12], noise, junction_state[12:15], junction_state[15:18])

                As.append((rf + DISCOUNT_VALUE * vf) - vp)
                grad_As.append(-As[-1] * (noise - junction_action) / (np.linalg.norm(noise - junction_action)))
                junction_states.append(junction_state)
                As[-1] = As[-1]**2
        print('Advantage gradients calculated.')

        # Optimize policy network
        print('Policy network training')
        with policy_sess.as_default():
            with policy_graph.as_default():
                pbar = tqdm(range(len(trajectories) - 1))
                pbar.set_description('Optimising policy network')

                param_update = 0

                for grad, state in zip(grad_As, junction_states):
                    pbar.update(1)

                    # Feed the data to the optimization graph
                    nk, beta= policy_sess.run([policy_net.train_op, policy_net.beta],
                                              feed_dict={policy_net.input: [normalize_state(state)], policy_net.action_grads: grad.reshape([1, 4])})

                    learningRate = min(23., beta[0][0])
                    # Add up nk
                    param_update += learningRate * nk[0]

                pbar.close()
                del(pbar)

                # Apply learning rate to new update step
                param_update = (1./(len(trajectories) - 1)) * np.array(param_update)
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

                policy_net.optimizer.apply_gradients(grads_and_vars).run()

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
        DroneInterface.release()

def run_test(arguments):
    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Instantiate publisher and subscriber for Gazebo
    DroneInterface.set_timestep(TIME_STEP)
    DroneInterface.init()
    traj = Trajectory()

    gui = GUI(0.046 * 50)
    frame_rate = 1 / 20.

    # Initialize policy network
    policy_net = PolicyNet(shape=[18,64,64,4])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if isinstance(arguments.test, str):
            saver.restore(sess, arguments.test)
        sess.run(tf.global_variables_initializer())

        for _ in range(1):
            TARGET = (np.random.randint(-2, 2), np.random.randint(-2, 2), 10)
            print('TARGET', TARGET)

            # Generate random start position for the Quadcopter
            traj.set_pose(DroneInterface.random_pose())
            _position = np.subtract(traj.get_pose()[4:7], TARGET)

            frame_time = 0

            for _ in range(1024):
                # Get state information from drone subscriber
                orientation, position, angular, linear = traj.get_state()

                # Calculate rotation matrix from quaternion
                orientation = Quaternion(orientation)

                state = {'position': position, 'rotation_matrix': orientation.rotation_matrix}

                if frame_time - frame_rate < time():
                    gui.update(state, 0)
                    frame_time = time()

                orientation = np.ndarray.flatten(orientation.rotation_matrix)

                # Calculate relative distance to target position
                position = np.subtract(position, TARGET)

                # Concatenate all to generate an input state vector for the networks
                state = np.concatenate((orientation, position/_position, angular/ANGULAR_VEL_NORM, linear/LINEAR_VEL_NORM))

                # Predict action with policy network
                action = np.array(sess.run(policy_net.model(), feed_dict={policy_net.input:[state]})[0])

                # Add bias to guarantee take off
                action = ACTION_SCALE * action + ACTION_BIAS

                #action = np.clip(action, 0, ACTION_MAX)

                # Feed action vector to the drone
                traj.step(action)
