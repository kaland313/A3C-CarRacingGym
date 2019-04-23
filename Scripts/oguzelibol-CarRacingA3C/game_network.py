#!/usr/bin/python
import tensorflow as tf
import numpy as np
import constants as Constants

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    self._device = device
    self._action_size = action_size

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])

      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

      # policy entropy
      entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.mul( log_pi, self.a ), reduction_indices=1 ) * self.td + entropy * entropy_beta )

      # R (input for value)
      self.r = tf.placeholder("float", [None])

      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, S):
    raise NotImplementedError()

  def run_policy(self, sess, S):
    raise NotImplementedError()

  def run_value(self, sess, S):
    raise NotImplementedError()

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
      #with tf.op_scope([], name, "GameACNetwork") as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  def _fc_variable(self, weight_shape):
    input_channels, output_channels = weight_shape[:2]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]

    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv_variable(self, weight_shape):
    w, h, input_channels, output_channels = weight_shape[:4]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]

    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, device)



    # Conv1 params
    conv1_input_depth    =  Constants.IMAGE_DEPTH

    # Conv2 params
    conv2_input_depth    =  Constants.CONV1_NUM_FILTERS

    dense_layer_input    =  1 + (((1 + (Constants.IMAGE_SIZE - Constants.CONV1_FILTER_SIZE) / Constants.CONV1_FILTER_STRIDE) \
                            - Constants.CONV2_FILTER_SIZE) / Constants.CONV2_FILTER_STRIDE)

    dense_layer_size     =  dense_layer_input * dense_layer_input * Constants.CONV2_NUM_FILTERS


    with tf.device(self._device):
      # filter height, width, in_channels, out_channels
      # 96-8 = 88/4 = 22, 22 +1 = 23
      self.W_conv1, self.b_conv1 = self._conv_variable([Constants.CONV1_FILTER_SIZE,
                                                        Constants.CONV1_FILTER_SIZE,
                                                        Constants.IMAGE_DEPTH,
                                                        Constants.CONV1_NUM_FILTERS])

      # 23-3 = 20, 20/2 = 10, 10+1= 11
      self.W_conv2, self.b_conv2 = self._conv_variable([Constants.CONV2_FILTER_SIZE,
                                                        Constants.CONV2_FILTER_SIZE,
                                                        Constants.CONV1_NUM_FILTERS,
                                                        Constants.CONV2_NUM_FILTERS])

      # 11x11x32 = 3872
      self.W_fc1, self.b_fc1 = self._fc_variable([dense_layer_size, Constants.DENSE_LAYER_INPUT_SIZE])

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([Constants.DENSE_LAYER_INPUT_SIZE, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([Constants.DENSE_LAYER_INPUT_SIZE, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, 4])

      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, Constants.CONV1_FILTER_STRIDE) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, Constants.CONV2_FILTER_STRIDE) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, dense_layer_size])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      # value (output)
      v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

  def run_policy_and_value(self, sess, S):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [S]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, S):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [S]} )
    return pi_out[0]

  def run_value(self, sess, S):
    v_out = sess.run( self.v, feed_dict = {self.s : [S]} )
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1,   self.b_fc1,
            self.W_fc2,   self.b_fc2,
            self.W_fc3,   self.b_fc3]