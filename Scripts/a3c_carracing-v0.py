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

import math
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

import constants as Constants

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
# Helper functions
#############################################################

def record(episode, episode_reward, worker_idx, global_ep_reward, result_queue, total_loss, num_steps, global_steps):
    """Helper function to store score and print statistics.

    :param episode: Current episode
    :param episode_reward: Reward accumulated over the current episode
    :param worker_idx: Which thread (worker)
    :param global_ep_reward: The moving average of the global reward
    :param result_queue: Queue storing the moving average of the scores
    :param total_loss: The total loss accumualted over the current episode
    :param num_steps: The number of steps the episode took to complete
    :param global_steps: The total number of steps taken by all workers
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
        'Episode: ' + str(episode) +' | ' +
        'Moving Average Reward: ' + str(int(global_ep_reward)) + ' | ' +
        'Episode Reward: ' +str(int(episode_reward)) + ' | ' +
        'Loss: ' + str(int(total_loss / float(num_steps) * 1000) / 1000) + ' | ' +
        'Steps: ' + str(num_steps) + ' | ' +
        'Worker: ' + str(worker_idx) + ' | ' +
        'Global steps: ' + str(global_steps)
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward

# log uniform
def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return math.exp(v)


#############################################################
# Random agent
#############################################################
class RandomAgent:
    """Random Agent that will play the specified game.

    :param env_name: Name of the environment to be played
    :param max_eps: Maximum number of episodes to run agent for.
    """

    def __init__(self, env_name, max_eps):
        self.env = gym.make(env_name)
        self.max_episodes = max_eps
        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def run(self):
        reward_avg = 0
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                _, reward, done, _ = self.env.step(self.env.action_space.sample())
                steps += 1
                reward_sum += reward
                self.env.render()
            # Record statistics
            self.global_moving_average_reward = record(episode,
                                                       reward_sum,
                                                       0,
                                                       self.global_moving_average_reward,
                                                       self.res_queue, 0, steps)

            reward_avg += reward_sum
        final_avg = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
        return final_avg


#############################################################
# Actor-critic model definition
#############################################################
class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.conv1 = layers.Conv2D(16, 8, strides=4, activation='relu', data_format="channels_last")
        self.conv2 = layers.Conv2D(32, 3, strides=2, activation='relu', data_format="channels_last")
        self.flatten = layers.Flatten()
        self.policy_logits = layers.Dense(action_size)
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        logits = self.policy_logits(x)
        values = self.values(x)
        return logits, values


#############################################################
# Master agent definition
#############################################################
class MasterAgent():
    def __init__(self):
        # Logging directory setup
        self.game_name = 'CarRacing-v0'
        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get input and output parameters and instantiate global network
        env = gym.make(self.game_name)
        print(self.game_name + " observation space shape: " + str(env.observation_space.shape))
        print(self.game_name + " action space shape: " + str(env.action_space.shape[0]))
        self.state_size = env.observation_space.shape
        # The game state converted to grayscale and Constants.IMAGE_DEPTH=4 are stacked to form the input to the network
        # Thus the original 3 sized 3rd axis (rgb) should have a size of 4
        # Note: + operator on touples acts as merge
        self.state_size = self.state_size[0:2]+(Constants.IMAGE_DEPTH,)
        self.action_size = Constants.ACTION_SIZE
        print("Network input space shape: ",  self.state_size)
        print("Network output ", Constants.ACTION_SIZE)
        self.actions = Constants.ACTIONS
        # Instantiate global network
        self.global_model = ActorCriticModel(self.state_size, self.action_size)
        # Evaluate global network with random input
        self.global_model(tf.convert_to_tensor(np.random.random(((1,) + self.state_size)), dtype=tf.float32))

        # Instantiate optimizer
        initial_learning_rate = log_uniform(Constants.ALPHA.LOW, Constants.ALPHA.HIGH, Constants.ALPHA.LOG_RATE)
        self.opt = tf.train.RMSPropOptimizer(initial_learning_rate, decay=Constants.RMSP.ALPHA,
                                             epsilon=Constants.RMSP.EPSILON, use_locking=True, centered=True)

    def train(self):
        if args.algorithm == 'random':
            random_agent = RandomAgent(self.game_name, args.max_eps)
            random_agent.run()
            return

        print("Starting training")
        res_queue = Queue()

        workers = [Worker(self.state_size, self.action_size, self.global_model, self.opt,
                          res_queue,
                          i, game_name=self.game_name,
                          save_dir=self.save_dir) for i in range(1)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        # workers = Worker(self.state_size, self.action_size, self.global_model, self.opt,
        #                  res_queue,
        #                  0, game_name=self.game_name,
        #                  save_dir=self.save_dir)
        #
        # print("Workers created, starting worker.")
        # workers.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break

        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,
                                 '{} Moving Average.png'.format(self.game_name)))
        plt.show()

    def play(self):
        env = gym.make(self.game_name)
        env_state = env.reset()
        state = processAndStackFrames(env_state)
        model = self.global_model
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                logits, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                action = np.argmax(probs)
                game_commands = self.actions[action]
                # Action softening based on action certainty
                game_commands = np.max(probs) * np.array(game_commands)
                new_frame, reward, done, _ = env.step(game_commands)
                env.render()
                processAndStackFrames(new_frame, state)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()


#############################################################
# Memory class
#############################################################
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


#############################################################
# Worker agent definition
#############################################################
class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()
    global_steps = 0

    def __init__(self, state_size, action_size, global_model, opt, result_queue, idx, game_name='CarRacing-v0',
                 save_dir='/tmp'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.actions = Constants.ACTIONS
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name)
        self.save_dir = save_dir
        self.ep_loss = 0.0
        self.maxEpReward = 0.0


    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < args.max_eps:
            # try:
            env_state = self.env.reset() #Returns: observation (object): the initial observation of the env
            self.env.render()
            # except:
            #     print("Exception while env.reset()")
            #     env_state = np.random.random(self.state_size)

            current_state = processAndStackFrames(env_state)
            mem.clear()
            ep_reward = 0.0
            self.maxEpReward = 0.0
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                logits, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                action = np.argmax(probs)

                game_commands = self.actions[action]

                # Action softening based on action certainty
                game_commands = np.max(probs)*np.array(game_commands)

                # action = np.random.choice(self.action_size, p=probs.numpy()[0])
                # new_state, reward, done, info = self.env.step(self.actions[action])
                new_state, reward, done, info = self.env.step(game_commands)
                new_state = processAndStackFrames(new_state, current_state)
                self.env.render()

                ep_reward += reward

                # Early termination
                if ep_reward > self.maxEpReward:
                    self.maxEpReward = ep_reward
                if self.maxEpReward - ep_reward > 5:
                    done = True

                # clip reward
                reward = np.clip(reward, -1, 1)

                mem.store(current_state, action, reward)

                if time_count == args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       args.gamma)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                 self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        Worker.global_steps += ep_steps
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward, self.result_queue,
                                   self.ep_loss, ep_steps, Worker.global_steps)
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, "
                                      "episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}.h5'.format(self.game_name))
                                )
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(np.stack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        actions_one_hot = tf.one_hot(memory.actions, self.action_size, dtype=tf.float32)

        policy = tf.nn.softmax(logits)
        entropy = tf.reduce_sum(policy * tf.log(policy + 1e-20), axis=1)

        policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot,
                                                                 logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


def processAndStackFrames(new_frame, current_state=None):
    """Converts a frame (state from the gym environment) and stacks appends it to the current_state, wich
    stores the previous 4 states of the environment.

    :param new_frame: The new state/frame from the environment.
    :param current_state: The state before stepping the environment and receiving a new frame (state).
    It contains the environment's state from the previous 4 steps.
    """
    gray_frame = rgb2gray(new_frame)

    if(current_state is not None):
        gray_frame = np.expand_dims(gray_frame, axis=2)
        new_state = np.append(current_state[:, :, 1:], gray_frame, axis=2)
    else:
        new_state = np.stack((gray_frame, gray_frame, gray_frame, gray_frame), axis=2)
    return new_state


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

