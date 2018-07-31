import os
import pickle
import numpy as np
import tensorflow as tf
from PID import PID
from tqdm import tqdm
from random import sample
from time import (sleep, time)
from pyquaternion import Quaternion
from gui import GUI
from constants import *
from utils import *
from drone_interface_icarus import DroneInterface
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

    interface = DroneInterface()
    interface.set_timestep(TIME_STEP)

    # Instantiate PID controllers for nominal orientation
    pitch_PID = PID(0, 0, 0)
    roll_PID = PID(0, 0, 0)
    yaw_PID = PID(0, 0, 0)

    pitch_PID = PID(0, 0, 0)
    roll_PID = PID(0, 0, 0)
    yaw_PID = PID(0, 0, 0)

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

    # Start main training loop
    for t in range(0, TRAJECTORIES_N):

        if arguments.log:
            policy_log = open('policy_loss.txt', 'a')
            value_log = open('value_loss.txt', 'a')

        # List for the trajectories of the current training cycle
        trajectories = []

        print('Trajectory #' + str(t+1))

        # Reset simulation and PID controller
        interface.reset()
        pitch_PID.clear()
        roll_PID.clear()
        yaw_PID.clear()

        # Generate random target position for current training cycle
        TARGET = (np.random.randint(-10, 10), np.random.randint(-10, 10), np.random.randint(2,10))
        TARGET = (np.random.randint(-2, 2), np.random.randint(-2, 2), 10)
        print('TARGET', TARGET)

        # Initialize random branch / junction points in time
        branches = sorted(sample(range(INITIAL_LENGTH), BRANCHES_N))
        branch_poses = []

        # Generate random start position for the Quadcopter
        interface.initial_pose()

        # Temporary lists
        states = []
        actions = []
        costs = []

        # Initial flight
        for b in range(INITIAL_LENGTH):
            # Get state information from drone subscriber
            orientation, position, angular, linear = interface.get_state()

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

            # Save prediction for the optimization step
            actions.append(action)

            # Calculate and save cost of state
            costs.append(cost(position, action, angular, linear))

            #if b < 20:
            #print(position)
            #    sleep(0.2)
            #elif b % 100 ==0:
            #    print(position)

            # Add PID controller outputs
            action[0] += -pitch_PID.output + yaw_PID.output
            action[1] += +roll_PID.output - yaw_PID.output
            action[2] += +pitch_PID.output + yaw_PID.output
            action[3] += -roll_PID.output - yaw_PID.output

            # Add bias to guarantee take off
            action = action + ACTION_BIAS

            # Clip output to guarantee a realistic simulation
            #action = np.clip(action, 0, ACTION_MAX)

            # Save full pose of the Quadcopter if a junction has to be created at this point of time later on
            if b in branches:
                branch_poses.append({'pose': interface.get_pose(), 'action': action, 'position': b})

            # Feed action vector to the drone
            interface.update(list(action))

            # Save state vector
            states.append(state)


        # Save initial trajectory to the list of the collection cycle
        trajectories.append({'level': -1, 'states': states, 'actions': actions, 'costs': costs})


        # Generate branch trajectories
        pbar = tqdm(range(len(branch_poses)*2))
        pbar.set_description('Generating branch trajectories')

        for i, b in enumerate(branch_poses):
            for n in range(0, NOISE_DEPTH):
                pbar.update(1)

                # Pause simulation and set starting pose of branch
                interface.set_pose(b['pose'])

                noise = None
                _noise = None
                states = []
                actions = []
                costs = []

                # Start branch trajectory
                for j in range(0, BRANCH_LENGTH):
                    # Get state information from drone subscriber
                    orientation, position, angular, linear = interface.get_state()

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

                    # Save prediction for the optimization step
                    actions.append(action)
                    _action = action

                    # Add PID controller outputs
                    action[0] += -pitch_PID.output + yaw_PID.output
                    action[1] += +roll_PID.output - yaw_PID.output
                    action[2] += +pitch_PID.output + yaw_PID.output
                    action[3] += -roll_PID.output - yaw_PID.output

                    # Add bias to guarantee take off
                    action = action + ACTION_BIAS

                    # Clip output to guarantee a realistic simulation
                    # action = np.clip(action, 0, ACTION_MAX)

                    if n == j:
                        # Generate noise if first state of branch
                        noise = NOISE()
                        _noise = noise
                        noise = noise + [ACTION_BIAS for _ in range(4)]
                        interface.update(list(noise))
                    elif j < n:
                        # Repeat noise for previous junction trajectory states
                        interface.update(list(trajectories[j-n]['noise']))
                    else:
                        # Feed action vector to the Quadcopter
                        interface.update(list(action))

                        # Save state vector
                        states.append(state)

                        # Calculate and save cost of state
                        costs.append(cost(position, _action, angular, linear))

                # Save branch trajectory to the list of the collection cycle
                trajectories.append({'level': n, 'noise': noise, '_noise': _noise, 'states': states, 'actions': actions, 'costs': costs, 'position': b['position']})
        pbar.close()
        del(pbar)

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
                    if not i % 3:
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

                pbar = tqdm(range(len(policy_traj) - 1))
                pbar.set_description('Optimising policy network')

                param_update = 0
                As = []

                # grads_and_vars_approx = []
                # grads_and_vars_dummy = []
                # epsilon = 0.01
                # for v in tf.trainable_variables():
                #     if len(v.shape) == 2:
                #         limit = v.shape[0].value * v.shape[1].value
                #     else:
                #         limit = v.shape[0].value
                #     grads_and_vars_approx.append(
                #         (np.reshape(np.float32(np.array([0. for _ in range(limit)])), v.shape), v))
                #     grads_and_vars_dummy.append(
                #         (np.reshape(np.float32(np.array([0. for _ in range(limit)])), v.shape), v))

                for i, trajectory in enumerate(policy_traj):
                    if(trajectory['level'] >= 0):
                        pbar.update(1)

                        # Get states of branch (off policy states)
                        states_f = trajectory['states']

                        # Get previous flight states (on policy states)
                        if trajectory['level'] == 0:
                            states_p = policy_traj[0]['states'][trajectory['position']:]
                        else:
                            states_p = policy_traj[i-1]['states']

                        T_f = len(states_f)
                        T_p = len(states_p)

                        # Predict value / reward of both trajectories termination state
                        with value_graph.as_default():
                            val_f = value_net.model().eval(session=value_sess, feed_dict={value_net.input:[states_f[T_f-1]]})[0][0]
                            val_p = value_net.model().eval(session=value_sess, feed_dict={value_net.input:[states_p[T_p-1]]})[0][0]

                        states = np.concatenate((states_f, states_p))
                        junction = states_p[0]
                        noise = trajectory['_noise']

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

                        # position = junction[9:12]
                        # angular = junction[12:15]
                        # linear = junction[15:18]
                        #
                        # rf = cost(position, noise, angular, linear)
                        # vf = 0
                        # for x, state in enumerate(states_f):
                        #     position = state[9:12]
                        #     angular = state[12:15]
                        #     linear = state[15:18]
                        #     action = np.array(policy_net.model().eval(session=policy_sess, feed_dict={policy_net.input: [state]})[0])
                        #
                        #     vf += (DISCOUNT_VALUE**(x+1)) * cost(position, action, angular, linear)
                        # vf += DISCOUNT_VALUE**(T_f+1)*val_f
                        #
                        # vp = 0
                        # for x, state in enumerate(states_p):
                        #     position = state[9:12]
                        #     angular = state[12:15]
                        #     linear = state[15:18]
                        #     action = np.array(policy_net.model().eval(session=policy_sess, feed_dict={policy_net.input: [state]})[0])
                        #     vp += DISCOUNT_VALUE**(x) * cost(position, action, angular, linear)
                        # vp += DISCOUNT_VALUE**(T_p)*val_p
                        #
                        # _A = (rf + vf) - vp
                        #
                        # action = np.array(
                        #     policy_net.model().eval(session=policy_sess, feed_dict={policy_net.input: [states_p[0]]})[0])
                        # action = (2 / 3.) * 10 ** (-5) * np.linalg.norm(action)
                        # vp = vp - action
                        #
                        # for i, grad_v in enumerate(grads_and_vars_dummy):
                        #     grad = grad_v[0]
                        #     v = grad_v[1]
                        #     if len(v.shape) == 2:
                        #         for x in range(v.shape[0].value):
                        #             for y in range(v.shape[1].value):
                        #                 grad[x, y] = epsilon
                        #                 policy_net.optimizer.apply_gradients(grads_and_vars_dummy)
                        #
                        #                 action = np.array(
                        #                     policy_net.model().eval(session=policy_sess,
                        #                                             feed_dict={policy_net.input: [states_p[0]]})[0])
                        #                 action = (2 / 3.) * 10 ** (-5) * np.linalg.norm(action)
                        #                 A_epsilon = (rf + vf) - (vp + action)
                        #
                        #                 grads_and_vars_approx[i][0][x, y] += (0.1/(len(policy_traj) - 1)) *(A_epsilon - _A) / epsilon
                        #
                        #                 grad[x, y] = -epsilon
                        #                 policy_net.optimizer.apply_gradients(grads_and_vars_dummy)
                        #                 grad[x, y] = 0
                        #     else:
                        #         for x in range(v.shape[0].value):
                        #             grad[x] = epsilon
                        #             policy_net.optimizer.apply_gradients(grads_and_vars_dummy)
                        #
                        #             action = np.array(
                        #                 policy_net.model().eval(session=policy_sess,
                        #                                         feed_dict={policy_net.input: [states_p[0]]})[0])
                        #             action = (2 / 3.) * 10 ** (-5) * np.linalg.norm(action)
                        #             A_epsilon = (rf + vf) - (vp + action)
                        #
                        #             grads_and_vars_approx[i][0][x] += (0.1/(len(policy_traj) - 1)) *(A_epsilon - _A) / epsilon
                        #
                        #             grad[x] = -epsilon
                        #             policy_net.optimizer.apply_gradients(grads_and_vars_dummy)
                        #             grad[x] = 0


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

                # for i, grad_v in enumerate(grads_and_vars):
                #     grad = grad_v[0]
                #     v = grad_v[1]
                #     grad2, v2 = grads_and_vars_approx[i]
                #     if len(v.shape) == 2:
                #         for x in range(v.shape[0].value):
                #             for y in range(v.shape[1].value):
                #                 print(grad[x, y], "  !=   ", grad2[x, y])
                #     else:
                #         for x in range(v.shape[0].value):
                #             print(grad[x], "  !=   ", grad2[x])

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

    # Instantiate publisher and subscriber for Gazebo
    interface = DroneInterface()
    interface.set_timestep(TIME_STEP)

    gui = GUI()

    # Initialize policy network
    policy_net = PolicyNet(shape=[18,64,64,4])
    saver = tf.train.Saver()

    pitch_PID = PID(0.5, 0., 10)
    roll_PID = PID(0.5, 0., 10)
    yaw_PID = PID(0, 0, 0)
    with tf.Session() as sess:
        saver.restore(sess, arguments.test)

        for _ in range(10):
            TARGET = (np.random.randint(-2, 2), np.random.randint(-2, 2), 10)
            print('TARGET', TARGET)

            pitch_PID.clear()
            roll_PID.clear()
            yaw_PID.clear()

            # Generate random start position for the Quadcopter
            interface.initial_pose()
            for _ in range(1024):
                # Get state information from drone subscriber
                orientation, position, angular, linear = interface.get_state()

                orientation = Quaternion(orientation)
                [yaw, pitch, roll] = orientation.yaw_pitch_roll
                pitch_PID.update(pitch)
                roll_PID.update(roll)
                yaw_PID.update(yaw)

                # Calculate relative distance to target position
                position = tuple(np.subtract(position, TARGET))

                # Calculate rotation matrix from quaternion
                orientation = Quaternion(orientation)
                orientation = orientation.rotation_matrix
                gui.update(orientation, position)
                orientation = np.ndarray.flatten(orientation)

                # Concatenate all tuples to generate an input state vector for the networks
                state = list(tuple(orientation) + position + angular + linear)

                # Predict action with policy network
                action = np.array(sess.run(policy_net.model(), feed_dict={policy_net.input:[state]})[0])

                #action = np.array([1000., 1000., 0., 0.])

                # Add PID controller outputs
                #action[0] += -pitch_PID.output
                #action[1] += +roll_PID.output
                #action[2] += +pitch_PID.output
                #action[3] += -roll_PID.output

                # Add bias to guarantee take off
                action = action +ACTION_BIAS

                # Clip output to guarantee a realistic simulation
                #action = np.clip(action, 0, ACTION_MAX)

                # Feed action vector to the drone
                interface.update(list(action))
                sleep(TIME_STEP*0.1)


# Value function as defined in formula 4
def value_function(sess, value_net, costs, states, i):
    values = []
    T = len(costs)
    value_factor = value_net.model().eval(session=sess, feed_dict={value_net.input:[states[T-1]]})[0][0]
    for t in range(i, T):
        v = DISCOUNT_VALUE**(t-i) * costs[t]
        values.append(v)
    return sum(values) + (DISCOUNT_VALUE**(T-i) * value_factor)

# Cost function as defined in formula 9
def cost(position, action, angular, linear):
    position = 4 * 10**(-3) * np.linalg.norm(position)
    action = (2/3.) * 10**(-5) * np.linalg.norm(action)
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

