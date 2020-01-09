from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf

from readers import reader
from modeling.models import builder
from modeling.utils import optimization
from modeling.utils import learning_rate_schedule

from protos import pipeline_pb2


def add_gradients_summaries(grads_and_vars):
  """Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
  Returns:
    The list of created summaries.
  """
  for grad, var in grads_and_vars:
    if grad is not None:
      tf.compat.v1.summary.histogram(
          'summarize_grads/' + var.op.name + '/gradient', grad)
      tf.compat.v1.summary.scalar(
          'summarize_grads/' + var.op.name + '/gradient_norm',
          tf.linalg.global_norm([grad]))
    else:
      logging.info('Var %s has no gradient', var.op.name)


def _create_model_fn(pipeline_proto, is_chief=True):
  """Creates a callable that build the model.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.

  Returns:
    A callable that takes [features, labels, mode, params] as inputs.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  def _model_fn(features, labels, mode, params):
    """Creates the model.

    Args:
      features: A dict mapping from names to tensors, denoting the features.
      labels: A dict mapping from names to tensors, denoting the labels.
      mode: Mode parameter required by the estimator.
      params: Additional parameters used for creating the model.

    Returns:
      An instance of EstimatorSpec.
    """
    is_training = (tf.estimator.ModeKeys.TRAIN == mode)
    logging.info("Current mode is %s, is_training=%s", mode, is_training)

    model = builder.build(pipeline_proto.model, is_training)

    # Predict resutls.
    predictions = model.predict(features)

    # Compute losses. Note: variables created in build_loss are not trainable.
    total_loss = 0
    losses = model.build_losses(features, predictions)
    for name, loss in losses.items():
      tf.compat.v1.summary.scalar('metrics/' + name, loss)
      total_loss += loss

    # Get variables_to_train.
    variables_to_train = model.get_variables_to_train()
    scaffold = model.get_scaffold()

    train_op = None
    eval_metric_ops = None

    if tf.estimator.ModeKeys.TRAIN == mode:
      global_step = tf.compat.v1.train.get_global_step()

      # Set learning rate.
      train_config = pipeline_proto.train_config
      lr_schedule_fn = learning_rate_schedule.create_learning_rate_schedule(
          train_config.learning_rate_schedule)
      learning_rate = lr_schedule_fn(global_step)
      tf.compat.v1.summary.scalar('metrics/learning_rate', learning_rate)

      # Use optimizer to minimize loss.
      optimizer = optimization.create_optimizer(train_config.optimizer,
                                                learning_rate=learning_rate)
      grad_and_vars = optimizer.compute_gradients(total_loss,
                                                  variables_to_train)
      add_gradients_summaries(grad_and_vars)

      train_op = optimizer.apply_gradients(grad_and_vars, global_step)

    elif tf.estimator.ModeKeys.EVAL == mode:

      eval_metric_ops = model.build_metrics(features, predictions)

    # Merge summaries.
    summary_saver_hook = tf.estimator.SummarySaverHook(
        summary_op=tf.compat.v1.summary.merge_all(),
        save_steps=pipeline_proto.train_config.save_summary_steps)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      training_hooks=[summary_saver_hook],
                                      scaffold=scaffold)

  return _model_fn


def train_and_evaluate(pipeline_proto, model_dir):
  """Starts the estimator trainval loop.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.
    model_dir: Path to the directory saving checkpoint files.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  # Create train_spec.
  train_config = pipeline_proto.train_config
  train_input_fn = reader.get_input_fn(pipeline_proto.train_reader,
                                       is_training=True)
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                      max_steps=train_config.max_steps)

  # Create eval_spec.
  eval_config = pipeline_proto.eval_config
  eval_input_fn = reader.get_input_fn(pipeline_proto.eval_reader,
                                      is_training=False)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=eval_config.steps,
      start_delay_secs=eval_config.start_delay_secs,
      throttle_secs=eval_config.throttle_secs)

  # Create run_config.
  run_config = tf.estimator.RunConfig(
      save_summary_steps=train_config.save_summary_steps,
      save_checkpoints_steps=train_config.save_checkpoints_steps,
      keep_checkpoint_max=train_config.keep_checkpoint_max,
      log_step_count_steps=train_config.log_step_count_steps)

  # Train and evaluate.
  model_fn = _create_model_fn(pipeline_proto, is_chief=run_config.is_chief)
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=model_dir,
                                     config=run_config)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
