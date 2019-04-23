#!/usr/bin/python
from collections import defaultdict
import tensorflow as tf

from tensorflow.python.training import slot_creator
from tensorflow.python.training import training_ops

class RMSPropApplier(object):

    def __init__(self, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, clip_norm=40.0, device="/cpu:0", name="RMSPropApplier"):

        self.name          =  name
        self.slots         =  defaultdict(dict)
        self.decay         =  decay
        self.device        =  device
        self.epsilon       =  epsilon
        self.momentum      =  momentum
        self.clipNorm      =  clip_norm
        self.learningRate  =  learning_rate

    def init(self):
        for name in ["learningRate", "decay", "momentum", "epsilon"]:
            self.__dict__[name + 'Tensor'] = tf.convert_to_tensor(self.__dict__[name], name=name)

    def getMakeSlot(self, var, slotName, opName, value=None, zeroesSlot=False):
        namedSlots = self.slots[slotName]
        if var not in namedSlots:
            namedSlots[var] = slot_creator.create_slot(var, value, opName) if not zeroesSlot else slot_creator.create_zeros_slot(var, opName)
        return namedSlots[var]

    def getSlot(self, var, name):
        slot = self.slots.get(name, None)
        if slot is None:
            return None
        return slot.get(var, None)

    def applyDense(self, grad, var):
        rms = self.getSlot(var, "rms")
        mom = self.getSlot(var, "momentum")
        return training_ops.apply_rms_prop(var, rms, mom, self.learningRateTensor, self.decayTensor, self.momentumTensor, self.epsilonTensor, grad, use_locking=False).op

    def apply_gradients(self, var_list, accum_grad_list, name=None):
        update_ops = []

        with tf.device(self.device):
            with tf.control_dependencies(None):
                for var in var_list:
                    value = tf.constant(1.0, dtype=var.dtype, shape=var.get_shape())
                    self.getMakeSlot(var, "rms", self.name, value)
                    self.getMakeSlot(var, "momentum", self.name, zeroesSlot=True)

            with tf.name_scope(name, self.name, []) as name:
                self.init()
                for var, accum_grad in zip(var_list, accum_grad_list):
                    with tf.name_scope("update_" + var.op.name), tf.device(var.device):
                        clipped_accum_grad = tf.clip_by_norm(accum_grad, self.clipNorm)
                        update_ops.append(self.applyDense(clipped_accum_grad, var))
                return tf.group(*update_ops, name=name)
