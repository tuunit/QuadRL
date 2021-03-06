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
import numpy as np
import multiprocessing
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from time import time
from termcolor import colored
from pyquaternion import Quaternion

import nn
from simulation import Trajectory
from simulation import PythonSimulator as Interface

from utils import Utils, Config
from visualizer import Visualizer

def run(arguments):
    if arguments.test:
        run_test(arguments)
    else:
        run_training(arguments)

def traj_step(args):
    traj, action = args
    traj.step(action)
    return traj

def visualize(trajectories):
    def state_to_dict(state):
        mat = [state[0:3], state[3:6], state[6:9]]
        return {'position': state[9:12], 'rotation_matrix': mat}

    visualizer = Visualizer(0.046 * 50)
    b = 1
    for i in range(len(trajectories[-1]['states']) + trajectories[-1]['position'] + trajectories[b]['level']):
        for j in range(b, len(trajectories)):
            if trajectories[b]['position'] + trajectories[b]['level'] + 1 == i:
                visualizer.add_quadrotor()
                b += 1
        for j in range(b):
            if j == 0:
                idx = i
            else:
                idx = i - (trajectories[j]['position'] + trajectories[j]['level'] + 1)
            if idx >= len(trajectories[j]['states']):
                continue

            visualizer.update(state_to_dict(trajectories[j]['states'][idx]), j)

def run_training(arguments):
    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Instantiate policy network with own Tensorflow graph
    policy_graph = tf.Graph()
    policy_sess = tf.Session(graph = policy_graph)

    policy_net = nn.PolicyNet(shape = Config.POLICY_SHAPE,
                              noise_cov = Config.NOISE_COV,
                              graph = policy_graph)

    with policy_sess.as_default():
        with policy_sess.graph.as_default():
            tf.global_variables_initializer().run()
            policy_net.saver = tf.train.Saver()
            if arguments.restore:
                policy_net.saver.restore(policy_sess,
                                        sorted(glob("tmp/policy_*"))[-1][:-5])

    # Instantiate value network with own Tensorflow graph
    value_graph = tf.Graph()
    value_sess = tf.Session(graph=value_graph)

    value_net = nn.ValueNet(shape = Config.VALUE_SHAPE,
                            graph = value_graph)

    with value_sess.as_default():
        with value_sess.graph.as_default():
            tf.global_variables_initializer().run()
            value_net.saver = tf.train.Saver()
            if arguments.restore:
                value_net.saver.restore(value_sess,
                                        sorted(glob("tmp/value_*"))[-1][:-5])

    if arguments.log:
        policy_log = open('policy_loss.txt', 'a')
        value_log = open('value_loss.txt', 'a')

    # Start main training loop
    for t in range(0, Config.TRAINING_EPOCHS):
        if arguments.log:
            policy_log = open('policy_loss.txt', 'a')
            value_log = open('value_loss.txt', 'a')

        # List for the trajectories of the current training cycle
        all_trajectories = []

        print('Trajectory #' + str(t+1))

        # Initialize random branch / junction points in time
        branches = sorted(np.random.randint(
                            0, Config.INITIAL_LENGTH, 
                            size=Config.BRANCHES_N))
        branch_indices = np.random.randint(
                            0, Config.INITIAL_N,
                            size=Config.BRANCHES_N)
        branch_trajs = {'trajs': [], 'positions': [], 'init_trajs': []}

        actions_sum = [0., 0., 0., 0.]
        actions_sum_abs = [0., 0., 0., 0.]
        actions_count = 0
        costs_sum = 0.
        costs_count = 0

        Interface.init()
        initial_trajs = []

        for _ in range(Config.INITIAL_N):
            tmp = Trajectory(Interface)
            initial_trajs.append(tmp)

        # Generate branch trajectories
        pbar = tqdm(range(Config.INITIAL_LENGTH))
        pbar.set_description('Generating Initial Trajectories')

        states_mat = []
        actions_mat = []
        costs_mat = []
        states = []
        for i, b in enumerate(initial_trajs):
            states.append(b.get_state())

        states = np.array(states)

        with multiprocessing.Pool(processes=4) as pool:
            for j in range(0, Config.INITIAL_LENGTH):
                pbar.update(1)

                actions = Utils.forward(policy_sess, policy_net, states)

                actions_sum += np.sum(actions, axis=0)
                actions_sum_abs += np.sum(np.abs(actions), axis=0)
                actions_count += Config.INITIAL_N

                actions_mat.append(actions)
                states_mat.append(states)

                actions_feed = Config.ACTION_SCALE * actions
                if j in branches:
                    for idx in range(branches.count(j)):
                        for _ in range(Config.NOISE_DEPTH):
                            branch = initial_trajs[branch_indices[branches.index(j) + idx]].snapshot()
                            branch_trajs['trajs'].append(branch)
                            branch_trajs['positions'].append(j)
                            branch_trajs['init_trajs'].append(branch_indices[branches.index(j) + idx])

                initial_trajs = pool.map(traj_step, zip(initial_trajs, actions_feed))
                states = np.array([traj.get_state() for traj in initial_trajs])
                costs_mat.append(Utils.compute_cost(states, actions))
                costs_sum += sum(costs_mat[-1])
                costs_count += len(costs_mat[-1])

        states_mat = np.array(states_mat)
        actions_mat = np.array(actions_mat)
        costs_mat = np.array(costs_mat)

        for i in range(len(initial_trajs)):
            actions = actions_mat[:, i]
            states = states_mat[:, i]
            costs = costs_mat[:, i]
            if len(actions) != len(states) or len(states) != len(costs) or len(states) != Config.INITIAL_LENGTH:
                print('ERROR: Anomalous trajectory data.')

            all_trajectories.append([{'level': -1, 'states': states, 'actions': actions, 'costs': costs}])
        pbar.close()
        del(pbar)

        # Generate branch trajectories
        pbar = tqdm(range(Config.BRANCH_LENGTH))
        pbar.set_description('Generating branch trajectories')

        states_mat = []
        actions_mat = []
        costs_mat = []
        noise_mat = []
        states = []
        for i, b in enumerate(branch_trajs['trajs']):
            states.append(b.get_state())

        states = np.array(states)

        with multiprocessing.Pool(processes=4) as pool:
            for j in range(0, Config.BRANCH_LENGTH):
                pbar.update(1)

                actions = Utils.forward(policy_sess, policy_net, states)

                actions_mat.append(actions)
                states_mat.append(states)

                if j < Config.NOISE_DEPTH:
                    mask = np.array([False if (x % Config.NOISE_DEPTH) >= j else True for x in range(Config.BRANCHES_N * Config.NOISE_DEPTH)])
                    noises = Utils.noise(Config.BRANCHES_N)
                    noises = np.repeat(noises, Config.NOISE_DEPTH, axis=0)
                    noises[mask] *= 0.

                    actions_feed = Config.ACTION_SCALE * (noises + actions)

                    mask = np.array([False if (x % Config.NOISE_DEPTH) == j else True for x in range(Config.BRANCHES_N * Config.NOISE_DEPTH)])
                    noises[mask] *= 0.
                    noise_mat.append(noises)
                else:
                    actions_feed = Config.ACTION_SCALE * actions

                branch_trajs['trajs'] = pool.map(traj_step, zip(branch_trajs['trajs'], actions_feed))

                states = np.array([traj.get_state() for traj in branch_trajs['trajs']])

                costs_mat.append(Utils.compute_cost(states, actions))

        noise_mat = np.array(noise_mat)
        states_mat = np.array(states_mat)
        actions_mat = np.array(actions_mat)
        costs_mat = np.array(costs_mat)

        for i, b in enumerate(branch_trajs['positions']):
            n = i % Config.NOISE_DEPTH

            noise = noise_mat[n, i]
            actions = actions_mat[n:, i]
            states = states_mat[n:, i]
            costs = costs_mat[n:, i]
            if len(actions) != len(states) or len(states) != len(costs) or len(states) != Config.BRANCH_LENGTH - (n):
                print('ERROR: Anomalous trajectory data.')

            all_trajectories[branch_trajs['init_trajs'][i]].append({'level': n, 'noise': noise, 'states': states, 'actions': actions, 'costs': costs, 'position': b})

        pbar.close()
        del(pbar)

        #visualize(all_trajectories[0])

        values_sum = 0.
        values_count = 0

        for value_traj in all_trajectories:
            # Value calculations
            terminal_states = [trajectory['states'][-1] for trajectory in value_traj]
            terminal_values = np.array([[0.] for _ in range (len(terminal_states))])
            terminal_values[0] = Utils.forward(value_sess, value_net, terminal_states)[0]

            for i, trajectory in enumerate(value_traj):
                trajectory['values'] = Utils.value_function_vectorized(trajectory['costs'], terminal_values[i][0])
            values_sum += value_traj[0]['values'][0]
            values_count += 1
            del terminal_values
            del terminal_states


        # Optimize value network
        pbar = tqdm(range(Config.VALUE_ITERATIONS))
        pbar.set_description('Optimizing value network')

        value_batch = None
        state_batch = None
        for trajectories in all_trajectories:
            for i, trajectory in enumerate(trajectories):
                if trajectory['level'] >= 0:
                    if value_batch is None:
                        value_batch = np.array(trajectory['values'][1:2])
                        state_batch = np.array(trajectory['states'][1:2])
                    else:
                        value_batch = np.concatenate((value_batch, trajectory['values'][1:2]))
                        state_batch = np.concatenate((state_batch, trajectory['states'][1:2]))
                    
                    if trajectory['level'] == 0:
                        value_batch = np.concatenate((value_batch, trajectories[0]['values'][trajectory['position']: trajectory['position'] + 1]))
                        state_batch = np.concatenate((state_batch, trajectories[0]['states'][trajectory['position']: trajectory['position'] + 1]))
        if value_batch.shape[0] != Config.BRANCHES_N * (Config.NOISE_DEPTH + 1):
            print("Invalid data for value network training")

        with value_sess.as_default():
            with value_sess.graph.as_default():
                for i in range(Config.VALUE_ITERATIONS):
                    pbar.update(1)
                    loss = value_net.train(value_sess,
                                           value_batch,
                                           Utils.normalize_states(state_batch))
                    if arguments.log:
                        value_log.write(str(loss)+'\n')
                    if not i % 99:
                        pbar.write("Value loss: {:.4f}".format(loss))
                    pbar.set_postfix(loss="{:.4f}".format(loss))
                    if loss <= Config.VALUE_LOSS_LIMIT:
                        break
        pbar.close()
        del(pbar)

        print('Mean Action    :', actions_sum / actions_count)
        print('Mean Action ABS:', actions_sum_abs / actions_count)
        print(colored('Average cost per time step: {}'.format(costs_sum / (costs_count*0.01)), 'blue'))

        if arguments.log:
            policy_log.write(str(costs_sum / (costs_count*0.01))+'\n')

        param_update = 0
        loss = 0

        pbar = tqdm(range(Config.BRANCHES_N*Config.NOISE_DEPTH))
        pbar.set_description('Optimising policy network')
        betas = 0

        As = []
        grad_As = []
        junction_states = []

        for trajectories in all_trajectories:
            # Advantages and their gradients w.r.t action computed here.

            for i, trajectory in enumerate(trajectories):
                if trajectory['level'] >= 0:
                    vf = trajectory['values'][1]
                    noise = trajectory['noise']

                    if trajectory['level'] == 0:
                        vp = trajectories[0]['values'][trajectory['position']]
                        junction_state = trajectories[0]['states'][trajectory['position']]
                        junction_action = trajectories[0]['actions'][trajectory['position']]
                    else:
                        vp = trajectories[i - 1]['values'][1]
                        junction_state = trajectories[i - 1]['states'][1]
                        junction_action = trajectories[i - 1]['actions'][1]

                    if len([i for i, j in zip(junction_state, trajectory['states'][0]) if abs(i - j) > 1e-4]) != 0:
                        print("Incorrect junction pairs calculated.")

                    rf = Utils.compute_cost([trajectory['states'][1]],
                                            [noise + junction_action])[0]

                    As.append((rf + Config.DISCOUNT_VALUE * vf) - vp)
                    grad_As.append(-As[-1] * noise / (np.linalg.norm(noise)))
                    junction_states.append(junction_state)

            loss += sum(As)

        # Optimize policy network
        #print('Policy network training')
        with policy_sess.as_default():
            with policy_graph.as_default():
                for grad, state in zip(grad_As, junction_states):
                    pbar.update(1)

                    # Feed the data to the optimization graph
                    nk, beta = policy_sess.run(
                                    [policy_net.train_op, 
                                     policy_net.beta],
                                    feed_dict = {
                                        policy_net.input: Utils.normalize_states([state]),
                                        policy_net.action_grads: grad.reshape([1, 4])
                                    })
                    beta = beta[0][0]
                    betas += beta
                    learningRate = min(2300., beta)
                    # Add up nk
                    param_update += learningRate * nk[0] / Config.BRANCHES_N

        pbar.close()
        del (pbar)


        with policy_sess.as_default():
            with policy_graph.as_default():
                # Apply learning rate to new update step
                param_update = -np.array(param_update)
                print('param_update:', param_update)
                print('betas:', betas / Config.BRANCHES_N)

                print('Policy Loss:', loss)

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
        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')

        timestamp = str(int(time()))
        with policy_sess.as_default():
            with policy_sess.graph.as_default():
                policy_net.saver.save(policy_sess, 'tmp/policy_checkpoint_' + timestamp + '.ckpt')

        with value_sess.as_default():
            with value_sess.graph.as_default():
                value_net.saver.save(value_sess, 'tmp/value_checkpoint_' + timestamp + '.ckpt')
        Interface.release()

def run_test(arguments):
    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Instantiate publisher and subscriber for Gazebo
    Interface.init()
    traj = Trajectory(Interface)

    visualizer = Visualizer(0.046 * 50)
    refresh_rate = 1 / 20.

    # Initialize policy network
    policy_net = nn.NeuralNet(shape = Config.POLICY_SHAPE)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if isinstance(arguments.test, str):
            saver.restore(sess, arguments.test)


        for _ in range(1):
            # Generate random start position for the Quadcopter
            traj.set_pose(Trajectory.random_pose())

            refresh_time = 0

            TARGETS = [[ 0,  0, 0],
                       [ 0,  0, 5],
                       [ 5,  5, 5],
                       [ 5, -5, 5],
                       [-5, -5, 5],
                       [-5,  5, 5],
                       [ 5,  5, 5],
                       [ 0,  0, 5],
                       [ 0,  0, 0]]
            c = 0

            positions = []
            for i in range(2500):
                if i % 250 == 0:
                    TARGET = TARGETS[c]
                    c += 1
                    print("TARGET: ", TARGET)
                # Get state information from drone subscriber
                state = traj.get_state()
                orientation = state[0:9]
                position = state[9:12]
                angular = state[12:15]
                linear = state[15:18]
                positions.append(position)

                # Calculate rotation matrix from quaternion
                orientation = Quaternion(matrix=np.reshape(orientation, (3,3)))

                state = {'position': position, 'rotation_matrix': orientation.rotation_matrix}

                if refresh_time - refresh_rate < time():
                    visualizer.update(state, 0)
                    visualizer.draw(positions)
                    visualizer.draw_target(TARGETS)
                    refresh_time = time()

                orientation = np.ndarray.flatten(orientation.rotation_matrix)

                # Calculate relative distance to target position
                #position = np.subtract(position, TARGET)

                position = np.subtract(position, [-0.29748929, -0.66516153, -0.43737027])
                position = np.subtract(position, TARGET)

                # Concatenate all to generate an input state vector for the networks
                state = np.concatenate((orientation, position, angular, linear))

                # Predict action with policy network
                action = Utils.forward(sess, policy_net, [state])[0]

                action = Config.ACTION_SCALE * action 

                # Feed action vector to the drone
                traj.step(action)
            input('Press Enter to exit')
