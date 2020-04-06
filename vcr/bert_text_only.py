from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json
import numpy as np
import tensorflow as tf

from protos import model_pb2
from modeling.layers import token_to_id
from modeling.utils import hyperparams
from vcr.model_base import ModelBase

from readers.vcr_fields import InputFields
from readers.vcr_fields import NUM_CHOICES

from bert.modeling import BertConfig
from bert.modeling import BertModel
from bert.modeling import get_assignment_map_from_checkpoint

import tf_slim as slim

FIELD_ANSWER_PREDICTION = 'answer_prediction'


class BertTextOnly(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(BertTextOnly, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.BertTextOnly):
      raise ValueError('Options has to be an BertTextOnly proto.')

    if model_proto.rationale_model:
      self._field_answer_choices = InputFields.rationale_choices_with_question
      self._field_answer_choices_len = InputFields.rationale_choices_with_question_len
      self._field_answer_label = InputFields.rationale_label
    else:
      self._field_answer_choices = InputFields.answer_choices_with_question
      self._field_answer_choices_len = InputFields.answer_choices_with_question_len
      self._field_answer_label = InputFields.answer_label

  def _predict_logits(self,
                      answer_choices,
                      answer_choices_len,
                      token_to_id_fn,
                      bert_config,
                      slim_fc_scope,
                      keep_prob=1.0,
                      is_training=False):
    """Predicts answer for a particular task.

    Args:
      answer_choices: A [batch, NUM_CHOICES, max_answer_len] string tensor.
      answer_choices_len: A [batch, NUM_CHOICES] int tensor.
      token_to_id_fn: A callable to convert the token tensor to an int tensor.
      slim_fc_scope: Slim FC scope.
      keep_prob: Keep probability of dropout layers.
      bert_config: A BertConfig instance to initialize BERT model.

    Returns:
      logits: A [batch, NUM_CHOICES] float tensor.
    """
    batch_size = answer_choices.shape[0]

    # Convert tokens into token ids.
    answer_choices_token_ids = token_to_id_fn(answer_choices)
    answer_choices_token_ids = tf.reshape(answer_choices_token_ids,
                                          [batch_size * NUM_CHOICES, -1])

    answer_choices_mask = tf.sequence_mask(answer_choices_len,
                                           maxlen=tf.shape(answer_choices)[-1])
    answer_choices_mask = tf.reshape(answer_choices_mask,
                                     [batch_size * NUM_CHOICES, -1])

    # Bert prediction.
    bert_model = BertModel(bert_config,
                           is_training,
                           input_ids=answer_choices_token_ids,
                           input_mask=answer_choices_mask)
    output = bert_model.get_pooled_output()

    # Classification layer.
    with slim.arg_scope(slim_fc_scope):
      output = slim.fully_connected(output,
                                    num_outputs=1,
                                    activation_fn=None,
                                    scope='logits')
    return tf.reshape(output, [batch_size, NUM_CHOICES])

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    token_to_id_layer = token_to_id.TokenToIdLayer(options.bert_vocab_file,
                                                   options.bert_unk_token_id)
    bert_config = BertConfig.from_json_file(options.bert_config_file)
    slim_fc_scope = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                  is_training)()

    # Prediction.
    answer_logits = self._predict_logits(inputs[self._field_answer_choices],
                                         inputs[self._field_answer_choices_len],
                                         token_to_id_layer, bert_config,
                                         slim_fc_scope,
                                         options.dropout_keep_prob, is_training)

    # Restore from checkpoint.
    assignment_map, _ = get_assignment_map_from_checkpoint(
        tf.global_variables(), options.bert_checkpoint_file)
    tf.compat.v1.train.init_from_checkpoint(options.bert_checkpoint_file,
                                            assignment_map)

    return {
        FIELD_ANSWER_PREDICTION: answer_logits,
    }

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    options = self._model_proto
    loss_fn = (tf.nn.sigmoid_cross_entropy_with_logits if options.use_sigmoid
               else tf.nn.softmax_cross_entropy_with_logits)

    labels = tf.one_hot(inputs[self._field_answer_label], NUM_CHOICES)
    losses = loss_fn(labels=labels, logits=predictions[FIELD_ANSWER_PREDICTION])

    return {'crossentropy': tf.reduce_mean(losses)}

  def build_metrics(self, inputs, predictions, **kwargs):
    """Compute evaluation metrics.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    accuracy_metric = tf.keras.metrics.Accuracy()
    y_true = inputs[self._field_answer_label]
    y_pred = tf.argmax(predictions[FIELD_ANSWER_PREDICTION], -1)

    accuracy_metric.update_state(y_true, y_pred)
    return {'metrics/accuracy': accuracy_metric}

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    options = self._model_proto
    trainable_variables = tf.compat.v1.trainable_variables()

    # Look for BERT frozen variables.
    frozen_variables = []
    for var in trainable_variables:
      for name_pattern in options.frozen_variable_patterns:
        if name_pattern in var.op.name:
          frozen_variables.append(var)
          break

    # Get trainable variables.
    var_list = list(set(trainable_variables) - set(frozen_variables))
    return var_list
