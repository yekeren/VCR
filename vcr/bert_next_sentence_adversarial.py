from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json
import numpy as np
import tensorflow as tf

from protos import model_pb2
from modeling.layers import token_to_id
from vcr.model_base import ModelBase

from readers.vcr_text_only_reader import InputFields
from readers.vcr_text_only_reader import NUM_CHOICES

from bert.modeling import BertConfig
from bert.modeling import BertModel
from bert.modeling import get_assignment_map_from_checkpoint

FIELD_ANSWER_PREDICTION = 'answer_prediction'
FIELD_ATTENTION_DIST = 'attention_dist'
MAX_BERT_LAYERS = 24


class VCRBertNextSentenceAdversarial(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VCRBertNextSentenceAdversarial,
          self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VCRBertNextSentenceAdversarial):
      raise ValueError(
          'Options has to be an VCRBertNextSentenceAdversarial proto.')

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    (answer_choices, answer_choices_len) = (
        inputs[InputFields.answer_choices_with_question],
        inputs[InputFields.answer_choices_with_question_len])
    batch_size = answer_choices.shape[0]

    # Convert tokens into token ids.
    token_to_id_layer = token_to_id.TokenToIdLayer(options.bert_vocab_file,
                                                   options.bert_unk_token_id)
    answer_choices_token_ids = token_to_id_layer(answer_choices)
    answer_choices_token_ids = tf.reshape(answer_choices_token_ids,
                                          [batch_size * NUM_CHOICES, -1])

    answer_choices_mask = tf.sequence_mask(answer_choices_len,
                                           maxlen=tf.shape(answer_choices)[-1])
    answer_choices_mask = tf.reshape(answer_choices_mask,
                                     [batch_size * NUM_CHOICES, -1])

    # Bert prediction.
    bert_config = BertConfig.from_json_file(options.bert_config_file)
    bert_model = BertModel(bert_config,
                           is_training,
                           input_ids=answer_choices_token_ids,
                           input_mask=answer_choices_mask)

    final_features = bert_model.get_pooled_output()

    # Classification layer.
    output = tf.compat.v1.layers.dense(final_features, units=1, activation=None)
    output = tf.reshape(output, [batch_size, NUM_CHOICES])

    # Restore from checkpoint.
    assignment_map, _ = get_assignment_map_from_checkpoint(
        tf.global_variables(), options.bert_checkpoint_file)

    tf.compat.v1.train.init_from_checkpoint(options.bert_checkpoint_file,
                                            assignment_map)

    # Adversarial training.
    attention_outputs = []
    for layer_id in range(MAX_BERT_LAYERS):
      tensor_name = 'bert/encoder/layer_%i/attention/self/Softmax:0' % layer_id
      attention_output = None
      try:
        attention_output = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
        attention_outputs.append(attention_output)
      except KeyError:
        logging.warn('Tensor %s doex not exist.', tensor_name)

    attention_outputs = tf.stack(attention_outputs, axis=0)
    attention_outputs = tf.reshape(
        tf.reduce_mean(attention_outputs, axis=[0, 2, 3]),
        [batch_size, NUM_CHOICES, -1])

    return {
        FIELD_ANSWER_PREDICTION: output,
        FIELD_ATTENTION_DIST: attention_outputs,
    }

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.one_hot(inputs[InputFields.answer_label], depth=NUM_CHOICES),
        logits=predictions[FIELD_ANSWER_PREDICTION])
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
    y_true = inputs[InputFields.answer_label]
    y_pred = tf.argmax(predictions[FIELD_ANSWER_PREDICTION], -1)

    accuracy_metric.update_state(y_true, y_pred)
    return {'metrics/accuracy': accuracy_metric}

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    options = self._model_proto

    bert_frozen_variables = []
    trainable_variables = tf.compat.v1.trainable_variables()
    if not options.bert_finetune_all:
      bert_frozen_variables = [
          x for x in trainable_variables if 'bert' in x.op.name
      ]
      for layer_name in options.bert_finetune_layers:
        bert_frozen_variables = [
            x for x in bert_frozen_variables if layer_name not in x.op.name
        ]

    var_list = [
        x for x in trainable_variables if not x in bert_frozen_variables
    ]
    return var_list