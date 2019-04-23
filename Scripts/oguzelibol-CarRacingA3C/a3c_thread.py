# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys
import gym

from game_state import GameState
from game_network import GameACFFNetwork

import constants as Constants

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
  def __init__(self, env, threadIndex, global_network, initialLearningRate, learningRateInput,
               grad_applier, maxGlobalTimeStep, saveData, device):

    self.env                 =  env
    self.saveData            =  saveData
    self.threadIndex         =  threadIndex
    self.learningRateInput   =  learningRateInput
    self.maxGlobalTimeStep   =  maxGlobalTimeStep
    self.local_network       =  GameACFFNetwork(Constants.ACTION_SIZE, device)

    self.local_network.prepare_loss(Constants.ENTROPY_BETA)
    with tf.device(device):
      var_refs = [v.ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(self.local_network.total_loss, var_refs, gate_gradients=False,
                                    aggregation_method=None, colocate_gradients_with_ops=False)

    self.apply_gradients      =  grad_applier.apply_gradients(global_network.get_vars(), self.gradients)
    self.sync                 =  self.local_network.sync_from(global_network)
    self.local_t              =  0
    self.maxEpReward          =  0
    self.prev_local_t         =  0
    self.episodeReward        =  0
    self.initialLearningRate  =  initialLearningRate

  def _anneal_learning_rate(self, globalTimeStep):
    return max(0, self.initialLearningRate * (self.maxGlobalTimeStep - globalTimeStep) / self.maxGlobalTimeStep)

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t, pi):
    summary_str = sess.run(summary_op, feed_dict={ score_input: score })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

    if self.threadIndex == 0:
      print('****** ADDING NEW SCORE ******')
      self.saveData.append(score, pi)
      if score > Constants.SAVE_SCORE_THRESHOLD:
        self.saveData.requestSave()

  def set_start_time(self, start_time):
    self.start_time = start_time

  def perfLog(self, global_t):
    if (self.threadIndex == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsedTime = time.time() - self.start_time
      stepsPerSec = global_t / elapsedTime
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsedTime, stepsPerSec, stepsPerSec * 3600 / 1000000.))

  def process(self, sess, global_t, summary_writer, summary_op, score_input):
    #ohe  - do these have to be self.states? or soemthing else?
    states   =  []
    actions  =  []
    rewards  =  []
    values   =  []

    terminal_end = False
    sess.run( self.sync )
    start_local_t = self.local_t

    # t_max times loop
    for i in range(Constants.LOCAL_T_MAX):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.S)
      action = self.choose_action(pi_)

      states.append(self.game_state.S)
      actions.append(action)
      values.append(value_)

      #change this to observe all the indices
      if (self.threadIndex == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("pi={}".format(pi_))
        print(" V={}".format(value_))

      # process game
      self.game_state.process(action)

      # receive game result
      reward    =  self.game_state.reward
      terminal  =  self.game_state.terminal

      self.episodeReward += reward

      #adding in early termination
      if self.episodeReward > self.maxEpReward:
        self.maxEpReward  = self.episodeReward

      if self.maxEpReward - self.episodeReward > 5:
        terminal = True

      # clip reward
      rewards.append( np.clip(reward, -1, 1) )

      self.local_t += 1
      self.game_state.update()

      if terminal:
        terminal_end = True
        print("score={}".format(self.episodeReward))

        self._record_score(sess, summary_writer, summary_op, score_input, self.episodeReward, global_t, pi_)

        self.maxEpReward    =  0 #ohe
        self.episodeReward  =  0

        self.game_state.reset()
        break

    R = 0.0 if terminal_end else self.local_network.run_value(sess, self.game_state.S)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si  =  []
    batch_a   =  []
    batch_td  =  []
    batch_R   =  []

    # compute and accmulate gradients
    for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R      =  ri + Constants.DISCOUNT * R
      td     =  R - Vi
      a      =  np.zeros([Constants.ACTION_SIZE])
      a[ai]  =  1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)
    sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s:   batch_si,
                  self.local_network.a:   batch_a,
                  self.local_network.td:  batch_td,
                  self.local_network.r:   batch_R,
                  self.learningRateInput: cur_learning_rate} )

    self.perfLog(global_t)

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t

