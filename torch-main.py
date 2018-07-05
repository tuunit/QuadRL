from drone_publisher import DronePublisher
from drone_subscriber import DroneSubscriber
from pyquaternion import Quaternion
from time import (sleep, time)
from random import sample
from PID import PID
import pickle

import rospy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, outputs, features=18, h1=64, h2=64):
        super(Net, self).__init__()
        self.h1 = nn.Linear(features,h1)
        self.h2 = nn.Linear(h1,h2)
        self.predict = nn.Linear(h2,outputs)

    def forward(self,x):
        x = F.tanh(self.h1(x))
        x = F.tanh(self.h2(x))
        x = self.predict(x)
        return x

# initialize ROS node for communication with Gazebo
rospy.init_node('hummingbird')

# constants
TRAJECTORIES_N = 256
BRANCHES_N = 32
TRAJ_INITIAL_LENGTH = BRANCHES_N * 5

NOISE_DEPTH = 2

VALUE_ITERATIONS = 1000
VALUE_LOSS_LIMIT = 0.0001

DISCOUNT_VALUE = 0.99

NOISE_COV = np.array(np.diag([600,600,600,600]), dtype=np.float32)
def NOISE(): return np.diag(np.diag(np.random.normal(80, 10, 4)) + NOISE_COV)

history = []

policy_fd = open('policy_loss_torch.txt', 'a')
value_fd = open('value_loss_torch.txt', 'a')

def main():
    # create publisher and subscriber
    pub = DronePublisher()
    sub = DroneSubscriber()

    pitch_PID = PID(80, 0, 20)
    roll_PID = PID(80, 0, 20)
    yaw_PID = PID(20, 0, 5)

    policy_net = Net(outputs=4)
    #policy_net.load_state_dict(torch.load('checkpoints/policy_checkpoint_1529066218.pth.tar'))
    #init.normal_(policy_net.predict.bias, mean=500, std=0.1)
    policy_net.double()
    
    value_net = Net(outputs=1)
    #value_net.load_state_dict(torch.load('checkpoints/value_checkpoint_1529066218.pth.tar'))
    value_net.double()
    value_net.optimizer = optim.SGD(value_net.parameters(), lr=0.01)
    value_net.criterion = nn.SmoothL1Loss()

    while not rospy.is_shutdown():
        # wait for rospy initialization
        sleep(0.1)

        # record trajectories
        print('Starting trajectory recording')

        for t in range(0, TRAJECTORIES_N):
            trajectories = []

            print('Trajectory #' + str(t))
            # Reset simulation
            pub.reset()

            TARGET = (np.random.randint(-20, 20), np.random.randint(-20, 20), np.random.randint(1,20))
            TARGET = (0, 0, 10)
            print 'TARGET', TARGET

            # Initialize branch positions
            branches = sorted(sample(range(TRAJ_INITIAL_LENGTH), BRANCHES_N))

            pitch_PID.clear()
            roll_PID.clear()
            yaw_PID.clear()

            pub.initial_pose()
            pub.unpause()

            states = []
            actions = []
            costs = []
            branch_poses = []

            # initial flight
            for b in range(TRAJ_INITIAL_LENGTH):
                sleep(0.01)
                # get state information from drone subscriber
                orientation, position, angular, linear = sub.get_state()

                # calculate relative distance to flight target
                position = tuple(np.subtract(position, TARGET))

                # calculate rotation matrix from quaternion
                orientation = Quaternion(orientation)
                [yaw, pitch, roll] = orientation.yaw_pitch_roll
                pitch_PID.update(pitch)
                roll_PID.update(roll)
                yaw_PID.update(yaw)

                orientation = np.ndarray.flatten(orientation.rotation_matrix)

                # concatenate all tuples to generate an input state vector for the networks
                state = list(tuple(orientation) + position + angular + linear)

                # get action vector from policy network prediction
                action = list(policy_net(torch.from_numpy(np.array(state))).data.numpy() + 459)
                action[0] += -pitch_PID.output + yaw_PID.output
                action[1] += +roll_PID.output - yaw_PID.output
                action[2] += +pitch_PID.output + yaw_PID.output
                action[3] += -roll_PID.output - yaw_PID.output
                action[0] = np.clip(action[0], 0, 900)
                action[1] = np.clip(action[1], 0, 900)
                action[2] = np.clip(action[2], 0, 900)
                action[3] = np.clip(action[3], 0, 900)

                if b in branches:
                    branch_poses.append({'pose': sub.get_pose(), 'action': action})

                # feed action vector to the drone
                pub.speed(action)

                # save state vector
                states.append(state)

                # save action vector
                actions.append(action)

                # calculate and save cost of state
                costs.append(cost(position, action, angular, linear))

            trajectories.append({'level': -1, 'states': states, 'actions': actions, 'costs': costs})

            print('Initial trajectory saved')

            # branch trajectories
            for i, b in enumerate(branch_poses):
                print('Trajectory #' + str(t) + ' | Branch #' + str(i))
                for n in range(0, NOISE_DEPTH):
                    pub.pause()
                    pub.set_pose(b['pose'])
                    pitch_PID.clear()
                    roll_PID.clear()
                    yaw_PID.clear()

                    sleep(0.01)
                    pub.unpause()

                    noise = None

                    states = []
                    actions = []
                    costs = []

                    for j in range(0, TRAJ_INITIAL_LENGTH / 2):
                        sleep(0.01)
                        # get state information from drone subscriber
                        orientation, position, angular, linear = sub.get_state()

                        # calculate relative distance to flight target
                        position = tuple(np.subtract(position, TARGET))

                        # calculate rotation matrix from quaternion
                        orientation = Quaternion(orientation)
                        [yaw, pitch, roll] = orientation.yaw_pitch_roll
                        pitch_PID.update(pitch)
                        roll_PID.update(roll)
                        yaw_PID.update(yaw)

                        orientation = np.ndarray.flatten(orientation.rotation_matrix)

                        # concatenate all tuples to generate an input state vector for the networks
                        state = list(tuple(orientation) + position + angular + linear)

                        # get action vector from policy network prediction
                        action = list(policy_net(torch.from_numpy(np.array(state))).data.numpy() + 459)
                        action[0] += -pitch_PID.output + yaw_PID.output
                        action[1] += +roll_PID.output - yaw_PID.output
                        action[2] += +pitch_PID.output + yaw_PID.output
                        action[3] += -roll_PID.output - yaw_PID.output
                        action[0] = np.clip(action[0], 0, 900)
                        action[1] = np.clip(action[1], 0, 900)
                        action[2] = np.clip(action[2], 0, 900)
                        action[3] = np.clip(action[3], 0, 900)

                        # feed action vector to the drone
                        if n == j:
                            noise = list(NOISE())
                            pub.speed(noise)
                        elif j < n:
                            pub.speed(trajectories[j-n]['noise'])
                        else:
                            pub.speed(action)

                        # save state vector
                        states.append(state)

                        # save action vector
                        actions.append(action)

                        # calculate and save cost of state
                        costs.append(cost(position, action, angular, linear))

                    trajectories.append({'level': n, 'noise': noise, 'states': states, 'actions': actions, 'costs': costs})
            pub.pause()

            with open('trajectories/torch_trajectory_' + str(t) + '.pkl', 'wb') as f:
                pickle.dump(trajectories, f)

            random_traj = get_random_trajectories()

            if random_traj:
                value_traj = trajectories + random_traj
            else:
                value_traj = trajectories

            print('Value network training')
            for i, trajectory in enumerate(value_traj):
                values = []
                for j in range(len(trajectory['states'])):
                    v = value_function(value_net, trajectory['costs'], trajectory['states'], j)
                    values.append(v)
                trajectory['values'] = values


            loss = 0
            for i in range(VALUE_ITERATIONS):
                loss = train_value_network(value_net, value_traj)
                value_fd.write(str(loss)+'\n')
                if not i % 250:
                    print("Value Loss: " + str(loss))
                if loss < VALUE_LOSS_LIMIT:
                    break;

            print('Policy network training')

            n_update = 0

            traj_len = len(trajectories)
            if random_traj:
                policy_traj = trajectories + random_traj
            else:
                policy_traj = trajectories

            skip = False
            As = []
            for i, trajectory in enumerate(policy_traj):
                if(trajectory['level'] >= 0):

                    # get states of branch (off policy states)
                    states_f = trajectory['states']
                    costs_f = trajectory['costs']

                    # get previous flight states (on policy states)
                    if trajectory['level'] == 1 and i < traj_len:
                        states_p = policy_traj[0]['states']
                        costs_p = policy_traj[0]['costs']
                    else:
                        states_p = policy_traj[i-1]['states']
                        costs_p = policy_traj[i-1]['costs']

                        if i >= traj_len and (i - traj_len) % 2 == 0:
                            continue

                    T_f = len(states_f)
                    T_p = len(states_p)
                    
                    states = np.concatenate((states_f, states_p))
                    junction = trajectory['states'][0]
                    noise = trajectory['noise']

                    action_pred = policy_net(torch.from_numpy(np.array(junction)))

                    J = []
                    action = [torch.tensor([action_pred[0], 0, 0, 0], dtype=torch.float64),
                         torch.tensor([0, action_pred[1], 0, 0], dtype=torch.float64),
                         torch.tensor([0, 0, action_pred[2], 0], dtype=torch.float64),
                         torch.tensor([0, 0, 0, action_pred[3]], dtype=torch.float64)]

                    for a in action:
                        params = np.array([])
                        policy_net.zero_grad()
                        action_pred.backward(a, retain_graph=True)
                        for param in policy_net.parameters():
                            p = np.ndarray.flatten(param.grad.data.numpy())
                            params = np.append(params, p)
                        J.append(params)
                    J = np.array(J)

                    U, s, V = np.linalg.svd(np.dot(np.linalg.cholesky(np.linalg.inv(NOISE_COV)), J), full_matrices=False)
                    h00_pinv = np.dot(np.dot(V.T, np.diag(np.square(1/s))), V)

                    position = junction[9:12]
                    angular = junction[12:15]
                    linear = junction[15:18]
                    
                    position = 4 * 10**(-3) * np.linalg.norm(position)
                    angular = 3 * 10**(-4) * np.linalg.norm(angular)
                    linear = 5 * 10**(-4) * np.linalg.norm(linear)
                    action = 2 * 10**(-4) * np.linalg.norm(noise)

                    rf = position + action + angular + linear

                    vf = value_function(value_net, costs_f, states_f, 1)
                    vp = value_function(value_net, costs_p, states_p, 0)

                    A = rf + DISCOUNT_VALUE * vf - vp 
                    As.append(A**2)

                    grads = [np.gradient(action_pred.data.numpy(), A)]
                    gk = np.dot(grads, J)
                    nk = np.dot(gk, h00_pinv)
                    n_update += nk

            loss = sum(As)

            policy_fd.write(str(loss)+'\n')
            print 'Policy Loss:', str(loss)

            n_update = (0.01/(len(policy_traj) - 1)) * np.array(n_update[0]) * 5

            with torch.no_grad():
                start = 0
                limit = 0
                for param in policy_net.parameters():
                    if len(param.shape) == 2:
                        limit = param.shape[0] * param.shape[1]
                    else:
                        limit = param.shape[0]
                    param -= torch.from_numpy(n_update[start:start+limit]).view(param.shape)
                    start = start + limit

            print n_update
            print policy_net.predict.bias

            timestamp = str(int(time()))
            torch.save(policy_net.state_dict(), 'checkpoints/policy_checkpoint_' + timestamp + '.pth.tar')
            torch.save(value_net.state_dict(), 'checkpoints/value_checkpoint_' + timestamp + '.pth.tar')
            history.append(trajectories)

# value function as defined in formula 4
def value_function(value_net, costs, states, i):
    values = []
    T = len(costs)
    value_factor = value_net(torch.tensor(states[T-1], dtype=torch.float64)).data.numpy()[0]
    for t in range(i, T):
        v = (DISCOUNT_VALUE**(t-i) * costs[t])
        values.append(v)
    return sum(values) + (DISCOUNT_VALUE**(T-i) * value_factor)

# cost function as defined in formula 9
def cost(position, action, angular, linear):
    position = 4 * 10**(-3) * np.linalg.norm(position)
    action = 2 * 10**(-4) * np.linalg.norm(action)
    angular = 3 * 10**(-4) * np.linalg.norm(angular)
    linear = 5 * 10**(-4) * np.linalg.norm(linear)

    return position + action + angular + linear

def train_value_network(value_net, value_traj):
    states = []
    values = []
    for t in value_traj:
        for s, v in zip(t['states'], t['values']):
            states.append(s)
            values.append(v)

    
    value_net.optimizer.zero_grad()
    output = value_net(torch.from_numpy(np.array(states)))
    target = torch.from_numpy(np.array(values)).view(-1, 1)

    loss = value_net.criterion(output, target)
    loss.backward()
    value_net.optimizer.step()

    return loss.data.numpy()

def get_random_trajectories():
    if not history:
        return None

    traj_n = sum([len(x) for x in history])
    amount = int(BRANCHES_N * NOISE_DEPTH)
    samples = sorted(sample(range(traj_n), traj_n if amount >= traj_n else amount))

    traj_samples = []
    count = 0
    for i in range(len(history)):
        for j in range(len(history[i])):
            if count in samples:
                traj_samples.append(history[i][j])

                if history[i][j]['level'] == 1:
                    traj_samples.append(history[i][0])
                elif history[i][j]['level'] > 1:
                    traj_samples.append(history[i][j-1])
            count = count + 1

    return traj_samples


if __name__ == '__main__':
    main()

