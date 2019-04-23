#!/usr/bin/python

import sys, gym
import numpy as np
import constants as Constants
from skimage.color import rgb2gray

class GameState(object):
  def __init__(self, env, display=False, no_op_max=7):
    self._no_op_max = no_op_max
    self.env = env
    self.actions = Constants.ACTIONS
    self.reset()

  def getProcessedFrame(self, observation):
    return rgb2gray(observation)

  def reset(self):
    self.reward = 0
    self.terminal = False

    X = self.env.reset()
    X = self.getProcessedFrame(X)
    self.S = np.stack((X, X, X, X), axis=2)

  def process(self, action):
    self.env.render()
    action = self.actions[action]

    X1, self.reward, self.terminal, info = self.env.step(action)
    X1 = self.getProcessedFrame(X1)
    X1 = np.expand_dims(X1, axis=2)
    self.S1 = np.append(self.S[:,:,1:], X1, axis=2)

  def update(self):
    self.S = self.S1
