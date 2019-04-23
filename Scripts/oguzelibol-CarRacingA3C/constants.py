#!/usr/bin/python

###############
# Game Config #
###############
GAME            =  'CarRacing-v0'
ACTION_ACCEL    =  [0, 1, 0]
ACTION_BRAKE    =  [0, 0, 0.8]
ACTION_LEFT     =  [-1, 0, 0]
ACTION_RIGHT    =  [1, 0, 0]
ACTIONS         =  [ACTION_ACCEL, ACTION_LEFT, ACTION_RIGHT, ACTION_BRAKE]
ACTION_SIZE     =  len(ACTIONS)


##################
# General Config #
##################
USE_GPU               =  False
LOCAL_T_MAX           =  5         				# repeat step size
NUM_THREADS           =  8         				# Number of parallel threads
SAVE_INTERVAL         =  120					# in seconds
SAVE_FILENAME         = 'scores-saved.json'
SAVE_SCORE_THRESHOLD  =  500


#####################
# Network constants #
#####################
IMAGE_SIZE              =  96
IMAGE_DEPTH             =  4

CONV1_FILTER_SIZE       =  8
CONV1_FILTER_STRIDE     =  4
CONV1_NUM_FILTERS       =  16

CONV2_FILTER_SIZE       =  3
CONV2_FILTER_STRIDE     =  2
CONV2_NUM_FILTERS       =  32

DENSE_LAYER_SIZE        =  3872
DENSE_LAYER_INPUT_SIZE  =  256


###################
# Alpha constants #
###################
class ALPHA:
    LOW         =  1e-4      # log_uniform low limit for learning rate
    HIGH        =  1e-2      # log_uniform high limit for learning rate
    LOG_RATE    =  0.4226    # log_uniform interpolate rate for learning rate (around 7 * 10^-4)


#####################
# RMSProp constants #
#####################
class RMSP:
    ALPHA       =  0.99      # decay parameter for RMSProp
    EPSILON     =  0.1       # epsilon parameter for RMSProp


########################
# Log & save constants #
########################
LOG_FILE        =  'tmp/a3c_log'
CHECKPOINT_DIR  =  'checkpoints'


###################
# Other Constants #
###################
GRADIENT_NORM_CLIP  =  40.0      # Gradient clipping norm
MAX_ITERATIONS      =  10 ** 6
ENTROPY_BETA        =  0.01      # Entropy Regularization Constant
DISCOUNT            =  0.99      # Discount



