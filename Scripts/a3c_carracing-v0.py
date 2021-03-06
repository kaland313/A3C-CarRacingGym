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
import time
import signal
import gym
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.color import rgb2gray

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers

import constants as Constants

import subprocess
import sys
from mpi4py import MPI

import csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

tf.enable_eager_execution()
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(sign=' ')

#############################################################
# Command line argument parser
#############################################################
parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'CarRacing-v0.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--load-model', dest='load_model', action='store_true',
                    help='Load a trained model for further training.')
parser.add_argument('--test', dest='test', action='store_true',
                    help='Test the trained model loaded from the saves directory.'
                         'Using multiple workers the model will be tested for 100 episodes. ')
parser.add_argument('--save-dir', default='../Outputs/', type=str,
                    help='Directory in which you desire to save the model.')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used for training, excluding the master managing the global model.')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--max-eps', default=5000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--beta', default=1e-4, type=float,
                    help='Entropy regularization coefficient beta.')
parser.add_argument('--save-threshold', default=300.0, type=float,
                    help='If a model is loaded, new one will overwrite it only if the achieved score is higher than this')

args = parser.parse_args()


#############################################################
# Helper functions
#############################################################
def record(episode, episode_reward, worker_idx, global_ep_reward, result_queue, total_loss, num_steps, global_steps,
           learning_rate, early_terminated, grad_norm, saved_value, csv_logger):
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

    elapsed_time = time.time() - start_time
    print(
        'Episode: ' + str(episode) + ' | ' +
        'Moving Avg Reward: ' + str(int(global_ep_reward)) + ' | ' +
        'Episode Reward: ' + str(int(episode_reward)) + ' | ' +
        'Loss: ' + str(int(total_loss / float(num_steps) * 1000) / 1000) + ' | ' +
        'Grad Norm: {:0.3f}'.format(float(grad_norm)) + ' | ' +
        'Value: {:0.3f}'.format(float(saved_value)) + ' | ' +
        'Steps: ' + str(num_steps) + ' | ' +
        'Worker: ' + str(worker_idx) + ' | ' +
        'Learning Rate: ' + str(learning_rate) + ' | ' +
        'Global steps: ' + str(global_steps) + ' | ' +
        'Time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + ' | ' +
        'Early Term: ' + str(early_terminated)
    )
    csv_logger.writerow(
        [str(episode), str(global_ep_reward), str(episode_reward), str(float(total_loss) / float(num_steps)),
         str(num_steps), str(global_steps), str(worker_idx), str(learning_rate),
         time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), str(early_terminated), str(float(grad_norm)),
         str(float(saved_value))])
    result_queue.put(global_ep_reward)
    return global_ep_reward


# log uniform
def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        if os.getuid() == 0:
            # Docker root user
            cmd = ["mpirun", "--allow-run-as-root", "-np", str(n), sys.executable] + ['-u'] + sys.argv
        else:
            # Non root user
            cmd = ["mpirun", "-np", str(n), sys.executable] + ['-u'] + sys.argv
        print(cmd)
        subprocess.check_call(cmd, env=env)
        return "parent"
    else:
        global nworkers, rank
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        print('assigning the rank and nworkers', nworkers, rank)
        return "child"


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
        self.conv1 = layers.Conv2D(16, 8, strides=4, activation='relu', data_format="channels_last",
                                   kernel_regularizer=regularizers.l2(0.01))
        self.conv2 = layers.Conv2D(32, 3, strides=2, activation='relu', data_format="channels_last",
                                   kernel_regularizer=regularizers.l2(0.01))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(Constants.DENSE_LAYER_INPUT_SIZE, activation='relu',
                                   kernel_regularizer=regularizers.l2(0.01))
        self.policy_logits = layers.Dense(action_size)
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.policy_logits(x)
        values = self.values(x)
        return logits, values


#############################################################
# Master agent definition
#############################################################
class MasterAgent():
    def __init__(self):
        self.game_name = Constants.GAME
        # Logging directory setup
        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log_file = open(save_dir + 'training_log.csv', 'w', newline='')
        self.log_writer = csv.writer(self.log_file, delimiter='\t')
        self.log_writer.writerow(['episode', 'global_ep_reward', 'episode_reward', 'loss', 'num_steps', 'global_steps',
                                  'worker_idx', 'learning_rate', 'elapsed_time', 'early_terminated', 'grad_norm',
                                  'saved_value'])
        self.log_file.flush()

        # Get input and output parameters and instantiate global network
        env = gym.make(self.game_name)
        print(self.game_name + " observation space shape: " + str(env.observation_space.shape))
        print(self.game_name + " action space shape: " + str(env.action_space.shape[0]))

        # The game state converted to grayscale and Constants.IMAGE_DEPTH=4 are stacked to form the input to the network
        # Thus the original 3 sized 3rd axis (rgb) should have a size of 4
        # Note: + operator on touples acts as merge
        self.state_size = env.observation_space.shape[0:2] + (Constants.IMAGE_DEPTH,)
        self.action_size = Constants.ACTION_SIZE
        print("Network input space shape: ", self.state_size)
        print("Network output ", Constants.ACTION_SIZE)

        self.actions = Constants.ACTIONS

        # Initialize global counters
        self.global_steps = 0
        self.global_episode = 0
        self.global_moving_average_reward = 0
        self.best_training_score = 0
        self.result_queue = Queue()
        self.moving_average_rewards = []  # record episode reward to plot

        # Instantiate global network
        self.global_model = ActorCriticModel(self.state_size, self.action_size)
        # To initialize the layers we evaluate global network with a random input
        self.global_model(tf.convert_to_tensor(np.random.random(((1,) + self.state_size)), dtype=tf.float32))

        # Instantiate optimizer
        # initial_learning_rate = log_uniform(Constants.ALPHA.LOW, Constants.ALPHA.HIGH, Constants.ALPHA.LOG_RATE)
        self.initial_learning_rate = args.lr
        self.opt = tf.train.RMSPropOptimizer(self.initial_learning_rate, use_locking=True, centered=False)

    def train(self):

        if args.algorithm == 'random':
            random_agent = RandomAgent(self.game_name, args.max_eps)
            random_agent.run()
            return

        print("Starting training")
        # res_queue = Queue()

        # Load a trained model
        if args.load_model:
            self.load_model()
            # self.play(False, "LoadedModel")

        # Broadcasting weights to workers
        m_send_packet = {'weights': self.global_model.get_weights()}
        comm.bcast(m_send_packet, root=0)

        # Receive packets from workers and update the weights
        while self.global_episode < args.max_eps:
            self.receive_and_update_weights()

        self.log_file.close()

        plt.clf()
        print("Moving average ep reward", self.moving_average_rewards)
        plt.plot(self.moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir, '{} Moving Average.png'.format(self.game_name)))
        plt.show()

    def play(self, load_model=True, video_title=""):
        model = self.global_model
        if load_model:
            self.load_model()

        env = gym.make(self.game_name)
        fig = plt.figure()
        step_counter = 0
        reward_sum = 0
        max_reward = 0
        ims = []
        done = False
        env_state = env.reset()
        state = processAndStackFrames(env_state)
        try:
            while not done:
                env.render()
                logits, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                action = np.argmax(probs)
                game_commands = self.actions[action]
                # Action softening based on action certainty
                game_commands = np.max(probs) * np.array(game_commands)
                new_frame, reward, done, _ = env.step(game_commands)
                state = processAndStackFrames(new_frame, state)
                # if step_counter > 80:
                #     plt.imshow(rgb2gray(new_frame),cmap='gray')
                #     plt.axis('off')
                #     plt.show()
                #     time.sleep(60)
                reward_sum += reward
                if max_reward < reward_sum:
                    max_reward = reward_sum
                discounted_reward_sum = reward + Constants.DISCOUNT * reward_sum

                # Early termination
                if max_reward - reward_sum > 15:
                    done = True

                print("Step: {:>4} | Reward: {:>4.1f} | ActionProbabilites: {} | ActionIdx: {}({}) | Steer,Gas,Break: {} "
                      .format(step_counter, reward_sum, np.array(probs)[0], action, Constants.ACTION_NAMES[action],
                              game_commands))
                ims.append([plt.imshow(new_frame, animated=True),
                            plt.text(3, 3, "V(s)={:0.3f}, R={:0.3f}".format(value[0, 0], discounted_reward_sum)),
                            plt.title(video_title)
                            ])
                step_counter += 1
            print("Final Reward: ", reward_sum, "Max reward: ", max_reward)
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")

        print("Saving gif file, please wait!")
        im_ani = animation.ArtistAnimation(fig, ims, blit=True)
        video_path = os.path.join(self.save_dir, '{}_{}.gif'.format(self.game_name, video_title))
        im_ani.save(video_path, writer='pillow', fps=24)

        env.close()

    def play_multi_worker(self):
        rewards = []
        max_rewards = []
        iii = 0
        while iii < 100:
            iii += 1
            m_recv_packet = comm.recv(source=MPI.ANY_SOURCE)
            rewards.append(m_recv_packet['reward_sum'])
            max_rewards.append(m_recv_packet['max_reward'])
            print("Episode", iii, "Final episode reward", m_recv_packet['reward_sum'],
                  "Max episode reward", m_recv_packet['max_reward'])
        print("Final episode reward ")
        print(rewards)
        print("Average ", np.mean(rewards), "+-", np.std(rewards))
        print("Max episode reward ")
        print(rewards)
        print("Average ", np.mean(max_rewards), "+-", np.std(max_rewards))

    def receive_and_update_weights(self):

        m_recv_packet = comm.recv(source=MPI.ANY_SOURCE)

        # Push gradients to global model
        self.opt.apply_gradients(zip(m_recv_packet['grads'], self.global_model.trainable_weights))

        # Reload the best model if the global_moving_average_reward collapsed
        if self.global_episode > 100 and self.global_moving_average_reward < (self.best_training_score - 375.0):
            self.load_model()
            print("Reloaded model, with best training score: ", self.best_training_score)
            self.log_writer.writerow(["Reloaded model, with best training score: " + str(self.best_training_score)])

        # Send updated weights to the worker
        m_send_packet = {'weights': self.global_model.get_weights(),
                         'global_episode': self.global_episode + 1}
        comm.send(m_send_packet, dest=m_recv_packet['worker_rank'])

        # Anneal learning rate
        self.opt._learning_rate = self._anneal_learning_rate(self.global_episode)

        if m_recv_packet['ep_done']:  # done and print information
            self.global_steps += m_recv_packet['ep_steps']
            self.global_moving_average_reward = record(self.global_episode,
                                                       m_recv_packet['ep_reward'],
                                                       m_recv_packet['worker_rank'],
                                                       self.global_moving_average_reward,
                                                       self.result_queue,
                                                       m_recv_packet['ep_loss'],
                                                       m_recv_packet['ep_steps'],
                                                       self.global_steps,
                                                       self.opt._learning_rate,
                                                       m_recv_packet['early_terminated'],
                                                       m_recv_packet['grad_norm'],
                                                       m_recv_packet['saved_value'],
                                                       self.log_writer)
            self.log_file.flush()
            self.moving_average_rewards.append(self.global_moving_average_reward)

            if self.global_moving_average_reward > self.best_training_score:
                print("Saving best model to {}, moving average reward: {}".format(self.save_dir,
                                                                                  self.global_moving_average_reward))
                self.global_model.save_weights(os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name)))
                self.best_training_score = self.global_moving_average_reward

            if self.global_episode % 500 == 0:
                self.play(False, "Ep" + str(self.global_episode))
            self.global_episode += 1

    def save(self):
        print("Saving manual model to {}, moving awerage reward: {}".
              format(self.save_dir, self.global_moving_average_reward))
        self.global_model.save_weights(os.path.join(self.save_dir, 'model_manual_{}.h5'.format(self.game_name)))

    def load_model(self):
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        self.global_model.load_weights(model_path)
        if int(self.best_training_score) == 0:
            # If load model was caled after startup, this will execute
            # Otherwise the best training score is either the value set here, or higher
            self.best_training_score = args.save_threshold

    def _anneal_learning_rate(self, global_ep):
        return max(0, self.initial_learning_rate * (args.max_eps - global_ep) / args.max_eps)


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
class Worker:
    def __init__(self, ):
        self.game_name = Constants.GAME
        self.env = gym.make(self.game_name)
        self.state_size = self.env.observation_space.shape[0:2] + (Constants.IMAGE_DEPTH,)
        self.action_size = Constants.ACTION_SIZE
        self.actions = Constants.ACTIONS
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        # To initialize the layers we evaluate global network with a random input
        action, value = self.local_model(tf.convert_to_tensor(np.random.random(((1,) + self.state_size)), dtype=tf.float32))
        self.ep_loss = 0.0
        self.maxEpReward = 0.0
        self.global_episode = 0

    def run(self):
        print("Starting worker", rank)
        # Receive global weights from master
        recv_packet = comm.bcast(None, root=0)
        # Initialize local model with new weights
        self.local_model.set_weights(recv_packet['weights'])
        print("Weights Received from master", rank)

        total_step = 1
        mem = Memory()
        while self.global_episode < args.max_eps:
            env_state = self.env.reset()  # Returns: observation (object): the initial observation of the env
            # self.env.render()

            current_state = processAndStackFrames(env_state)
            mem.clear()
            ep_reward = 0.0
            self.maxEpReward = 0.0
            ep_steps = 0
            self.ep_loss = 0
            early_terminated = False

            time_count = 0
            done = False
            while not done:
                logits, value = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                action = np.argmax(probs)

                # Random action based on the probabilities to force exploration
                # action = np.random.choice(self.action_size, p=probs.numpy()[0])

                game_commands = self.actions[action]

                # Action softening based on action certainty
                game_commands = probs.numpy()[0, action] * np.array(game_commands)

                new_state, reward, done, info = self.env.step(game_commands)
                new_state = processAndStackFrames(new_state, current_state)
                self.env.render()

                ep_reward += reward

                # Early termination
                if ep_reward > self.maxEpReward:
                    self.maxEpReward = ep_reward
                # if self.maxEpReward - ep_reward > 5:
                #     done = True
                #     early_terminated = True

                # # clip reward
                # reward = np.clip(reward, -1, 1)

                mem.store(current_state, action, reward)

                if ep_steps == 0:
                    saved_value = value

                if time_count == Constants.LOCAL_T_MAX or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done, new_state, mem, Constants.DISCOUNT)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)

                    # Gradient clipping
                    clipped_grads, global_norm = tf.clip_by_global_norm(grads, Constants.GRADIENT_NORM_CLIP)

                    send_packet = {'worker_rank': rank,
                                   'grads': clipped_grads,
                                   'grad_norm': global_norm,
                                   'ep_done': done,
                                   'ep_reward': ep_reward,
                                   'ep_loss': self.ep_loss,
                                   'ep_steps': ep_steps,
                                   'saved_value': saved_value,
                                   'early_terminated': early_terminated,
                                   }
                    # Send episode data the master process
                    comm.send(send_packet, dest=0)

                    # Receive new weights from the master process
                    recv_packet = comm.recv(source=0)
                    new_weights = recv_packet['weights']
                    self.global_episode = recv_packet['global_episode']

                    # Update local model with new weights
                    self.local_model.set_weights(new_weights)

                    mem.clear()
                    time_count = 0

                ep_steps += 1
                time_count += 1
                current_state = new_state
                total_step += 1

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            # terminal thus we can't expect any more reward form the preceeding states as there are none
            reward_sum = 0.
        else:
            _, reward_sum = self.local_model(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))
            reward_sum = float(reward_sum)

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(tf.convert_to_tensor(np.stack(memory.states), dtype=tf.float32))

        # Calculate advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values

        # Calculate value loss
        value_loss = tf.keras.losses.mean_squared_error(discounted_rewards, values)

        # Calculate policy loss
        actions_one_hot = tf.one_hot(memory.actions, self.action_size, dtype=tf.float32)
        policy = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(policy * tf.log(policy + 1e-20), axis=1)
        policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot, logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= args.beta * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

        return total_loss

    def play(self):
        model = self.local_model
        model_path = os.path.join(args.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)

        env = gym.make(self.game_name)
        iii = 0

        while iii < 100/args.workers + 1:
            iii += 1
            step_counter = 0
            reward_sum = 0
            max_reward = 0
            done = False
            env_state = env.reset()
            state = processAndStackFrames(env_state)
            try:
                while not done:
                    env.render()
                    logits, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                    probs = tf.nn.softmax(logits)
                    action = np.argmax(probs)
                    game_commands = self.actions[action]
                    # Action softening based on action certainty
                    game_commands = np.max(probs) * np.array(game_commands)
                    new_frame, reward, done, _ = env.step(game_commands)
                    state = processAndStackFrames(new_frame, state)

                    reward_sum += reward
                    if max_reward < reward_sum:
                        max_reward = reward_sum

                    # # Early termination
                    # if max_reward - reward_sum > 15:
                    #     done = True

                    step_counter += 1

                # Send episode data the master process
                send_packet = {'reward_sum': reward_sum,
                               'max_reward': max_reward}
                comm.send(send_packet, dest=0)
                # print("Final Reward: ", reward_sum, "Max reward: ", max_reward)
            except KeyboardInterrupt:
                print("Received Keyboard Interrupt. Shutting down.")
        env.close()


def processAndStackFrames(new_frame, current_state=None):
    """Converts a frame (state from the gym environment) and stacks appends it to the current_state, wich
    stores the previous 4 states of the environment.

    :param new_frame: The new state/frame from the environment.
    :param current_state: The state before stepping the environment and receiving a new frame (state).
    It contains the environment's state from the previous 4 steps.
    """
    gray_frame = rgb2gray(new_frame)
    if current_state is not None:
        gray_frame = np.expand_dims(gray_frame, axis=2)
        new_state = np.append(current_state[:, :, 1:], gray_frame, axis=2)
    else:
        new_state = np.stack((gray_frame, gray_frame, gray_frame, gray_frame), axis=2)
    return new_state


#############################################################
# The main
#############################################################
if __name__ == '__main__':
    if args.train and args.test:
        parser.error('Train and test cannot be set at the same time. '
                     'If you would like to run a test after training you have to do it manually.')

    if rank == 0:
        print(args)

        # Install Ctrl+C signal handler
        def signal_handler(sig, frame):
            print('CTRL+C was pressed, attempting to stop and save.')
            master.save()
        signal.signal(signal.SIGINT, signal_handler)

        if args.algorithm == 'random':
            randomAgent = RandomAgent('CarRacing-v0', 1)
            randomAgent.run()
            exit()
        elif args.train:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
                f.write(str(args))

            print("=Training=====================================================")
            print("Starting mpi processes")
            mpi_fork(args.workers + 1)
            master = MasterAgent()
            start_time = time.time()
            master.train()
        elif args.test:
            print("=Testing======================================================")
            print("Starting mpi processes")
            mpi_fork(args.workers + 1)
            master = MasterAgent()
            master.play_multi_worker()
        else:
            print("=Demo=========================================================")
            master = MasterAgent()
            master.play()

    else:
        worker = Worker()
        if args.train:
            worker.run()
        elif args.test:
            worker.play()
        # If the workers has nothing to do, they shouldn't do anything --> no default behaviour or else branch

