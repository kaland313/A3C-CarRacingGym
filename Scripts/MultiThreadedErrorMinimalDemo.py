import os

# Call XInitThreads as the _very_ first thing.
# After some Qt import, it's too late
import ctypes
import sys
if sys.platform.startswith('linux'):
    try:
        x11 = ctypes.cdll.LoadLibrary('libX11.so')
        x11.XInitThreads()
    except:
        print("Warning: failed to XInitThreads()")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

ACTION_ACCEL = [0, 1, 0]
ACTION_BRAKE = [0, 0, 0.8]
ACTION_LEFT  = [-1, 0, 0]
ACTION_RIGHT = [1, 0, 0]
ACTIONS      = [ACTION_ACCEL, ACTION_LEFT, ACTION_RIGHT, ACTION_BRAKE]
ACTION_SIZE  = len(ACTIONS)

tf.enable_eager_execution()

#############################################################
# Command line argument parser
#############################################################
parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'CarRacing-v0.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='../Outputs/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()


#############################################################
# Master agent definition
#############################################################
class MasterAgent():
    def __init__(self):
        # Logging directory setup
        self.game_name = 'CarRacing-v0'

    def train(self):
        print("Starting training")
        workers = [Worker(i, game_name=self.game_name) for i in range(multiprocessing.cpu_count())]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()


#############################################################
# Worker agent definition
#############################################################
class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    save_lock = threading.Lock()

    def __init__(self, idx, game_name='CarRacing-v0'):
        super(Worker, self).__init__()
        self.worker_idx = idx
        self.game_name = game_name


    def run(self):
        env = gym.make(self.game_name)
        while Worker.global_episode < args.max_eps:
            # try:
            env.reset() #Returns: observation (object): the initial observation of the env
            # except:
            #     print("Exception while env.reset()")
            #     env_state = np.random.random(self.state_size)
            time_count = 0
            done = False
            while not done:
                new_state, reward, done, info = env.step(env.action_space.sample())
                if time_count == args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    if done:  # done and print information
                        Worker.global_episode += 1
                time_count += 1

#############################################################
# The main
#############################################################
if __name__ == '__main__':
    print(args)
    # randomAgent = RandomAgent('CarRacing-v0', 4000)
    # randomAgent.run()
    master = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()
