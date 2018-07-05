# Quadcopter Reinforcement Learning

For this thesis, a Reinforcement Learning system for learning to fly
a Quadcopter without human interaction was implemented. The implementation is
split into the exploration and exploitation strategy. Meaning, that trajectories
are generated autonomously without human interaction and two neural networks,
a policy and a value network respectively, are trained on this data to predict
the thrust of the Quadcopter rotors needed to fly towards a target position.

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes on
how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
RotorS Simulator: https://github.com/ethz-asl/rotors_simulator
Python librarys:
- tensorflow
- numpy
- tqdm
- pyquaternion
- future

pip install -U -r requirements.txt

```

## Running the training

Start Gazebo / RotorS with the Hummingbird model.

```
roslaunch rotors_gazebo mav.launch mav_name:=hummingbird world_name:=basic
```

Start the training procedure by calling:

```
python main.py
```

## Running a test

Start a test run:

```
python main.py --test [path to policy network checkpoint]
```

## File Structure
- main.py (startup)
- qrl.py (main component / exploration and exploitation strategy implementation)
- neural_network.py (general purpose fully connected neural network generator)
- policy_network.py (neural network extension with optimization strategy of the policy network)
- value_network.py (neural network extension with optimization strategy of the value network)
- constants.py (important constants for the program flow)
- utils.py (some utility functions for the program)
- drone_subscriber.py (rospy subscriber class to get data from the simulator)
- drone_publisher.py (rospy publisher class to control and send data to the simulator)
- PID.py (PID controller implementation)
- torch_main.py (implementation of the system in PyTorch for test purposes)
