from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from protos import hyperparams_pb2
from protos import optimizer_pb2


def create_optimizer(options, learning_rate=0.1):
  """Builds optimizer from options.

  Args:
    options: An instance of optimizer_pb2.Optimizer.
    learning_rate: A scalar tensor denoting the learning rate.

  Returns:
    A tensorflow optimizer instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, optimizer_pb2.Optimizer):
    raise ValueError('The options has to be an instance of Optimizer.')

  optimizer = options.WhichOneof('optimizer')

  if 'adagrad' == optimizer:
    options = options.adagrad
    return tf.keras.optimizers.Adagrad(
        learning_rate,
        initial_accumulator_value=options.initial_accumulator_value,
        epsilon=options.epsilon)

  if 'rmsprop' == optimizer:
    options = options.rmsprop
    return tf.keras.optimizers.RMSprop(
        learning_rate,
        rho=options.rho,
        momentum=options.momentum,
        epsilon=options.epsilon,
        centered=options.centered)

  if 'adam' == optimizer:
    options = options.adam
    return tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=options.beta_1,
        beta_2=options.beta_2,
        epsilon=options.epsilon,
        amsgrad=options.amsgrad)

  raise ValueError('Invalid optimizer: {}.'.format(optimizer))
