#!/usr/bin/python
import qrl
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Quadcopter Reinforcment Learning')
parser.add_argument('--log', action='store_true', help='Log policy and value network losses')
parser.add_argument('--test', nargs='?', metavar='FILE PATH', default=False, const=True, type=str, help='Test policy network')

if __name__ == '__main__':
    qrl.run(parser.parse_args())
