#!/usr/bin/python
import tensorflow as tf
import threading
import numpy as np

import gym

import signal
import random
import math
import os
import time

from game_network import GameACFFNetwork
from a3c_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier
from game_state import GameState
from saver import Saver

import constants as Constants

# log uniform
def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return math.exp(v)

class Data:
    a3c_threads    =  None
    stop_requested =  False

def main():
    # Initialize variables
    device = "/gpu:0" if Constants.USE_GPU else "/cpu:0"
    learning_rate_input = tf.placeholder("float")
    initial_learning_rate = log_uniform(Constants.ALPHA.LOW, Constants.ALPHA.HIGH, Constants.ALPHA.LOG_RATE)

    # Initialize network
    global_network, grad_applier = init_network(device, learning_rate_input)

    # Load our saver class
    saver = Saver()

    # Initialize threads
    Data.a3c_threads = create_a3c_threads(global_network, learning_rate_input, initial_learning_rate, grad_applier, saver.data, device)

    # Initialize TF
    session = init_tf()
    summary = init_tf_summary(session)

    # Initialize / Load existing checkpoint
    saver.load(session)

    # Install Ctrl+C signal handler
    def signal_handler(sig, frame):
        print('CTRL+C was pressed, attempting to stop and save.')
        Data.stop_requested = True
    signal.signal(signal.SIGINT, signal_handler)

    # Create and trigger the threads
    threads = run_system_threads(session, Data, summary, saver)
    print('Press Ctrl+C to stop')

    # Finish and save results
    finish(threads)
    saver.save(session)



# Initialize and return the game network
def init_network(device, learning_rate_input):
    network  = GameACFFNetwork(Constants.ACTION_SIZE, device)

    grad_applier = RMSPropApplier(learning_rate  =  learning_rate_input,
                                  decay          =  Constants.RMSP.ALPHA,
                                  epsilon        =  Constants.RMSP.EPSILON,
                                  clip_norm      =  Constants.GRADIENT_NORM_CLIP,
                                  momentum       =  0.0,
                                  device         =  device)

    return network, grad_applier


def init_tf():
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                               allow_soft_placement=True))
    init = tf.initialize_all_variables()
    session.run(init)

    return session


def create_a3c_threads(network, learning_rate_input, initial_learning_rate, grad_applier, savedData, device):
    threads = []
    for i in range(Constants.NUM_THREADS):
        thread = A3CTrainingThread(None, i, network, initial_learning_rate, learning_rate_input, grad_applier, Constants.MAX_ITERATIONS, savedData, device = device)
        threads.append(thread)

    return threads


def run_system_threads(session, data, summary, saver):
    threads = []
    for i in range(Constants.NUM_THREADS):
        env = gym.make(Constants.GAME)
        threads.append(threading.Thread(target=train_function, args=(env, i, session, data, summary, saver)))

    saver.data.start_time = time.time() - saver.data.wall_t
    for t in threads:
        t.start()

    return threads

def init_tf_summary(session):
    score_input = tf.placeholder(tf.int32)
    tf.scalar_summary("score", score_input)
    summary_op      =  tf.merge_all_summaries()
    summary_writer  =  tf.train.SummaryWriter(Constants.LOG_FILE, session.graph)
    return summary_writer, summary_op, score_input


def finish(threads):
    signal.pause()
    print('Please wait while data is being saved')

    # Wait for threads to complete
    for t in threads:
        t.join()


def train_function(env, i, session, data, summary, saver):
    # Get the current thread object and attach the game env and state to it
    a3cthread = data.a3c_threads[i]
    a3cthread.env = env
    a3cthread.game_state = GameState(env)

    # Set the start timer for the current thread
    start_time = time.time() - saver.data.wall_t
    a3cthread.set_start_time(start_time)

    while True:
        # If Ctrl+C was pressed or we exceed the max time, break out of this loop
        if data.stop_requested or saver.data.global_t > Constants.MAX_ITERATIONS:
            break

        # Run this thread one step
        diff_global_t = a3cthread.process(session, saver.data.global_t, *summary)
        saver.data.global_t += diff_global_t

        saver.saveIfRequested(session)


if __name__ == "__main__":
    main()

