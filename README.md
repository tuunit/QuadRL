# QuadRL - Quadcopter Reinforcement Learning

This Reinforcement Learning system trains a neural network with the SARSA Critic-Actor strategy to perform waypoint tracking.

The system was implemented originally implemented during my thesis.Further improvements and changes were made ever since. The exploration and exploitation strategies are heavily influenced by a research paper from the ETH Zurich. [[1]](https://arxiv.org/abs/1707.05110)
The simulation interface was written genericly to make it easy to use different implementation. The pure python simulator published in this repository is a reimplementation of the RAI Quadcopter simulation from the ETH.


## Important Information
This is a work in progress, so you still have to wait for a complete documentation and additional information.

## Getting Started

If you want to try out the RL system, you just have to clone this repository, use Python3 and install the required python libraries with pip. (Optional: Setup up a python environment with virtualenv or anaconda)


```bash
pip install -U -r requirements.txt

```

### Start training process
```bash
python main.py
```
This will start the training process and save checkpoints to the folder `tmp`.


### Visualize a waypoint tracking scenario
```bash
python main.py --test tmp/policy_checkpoint_{latest_unix_timestamp}.ckpt
```
