from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json
import numpy as np
import tensorflow as tf

from protos import model_pb2
from modeling.layers import token_to_id
from modeling.models.model_base import ModelBase

from readers.vcr_reader import InputFields
from readers.vcr_reader import NUM_CHOICES

from official.nlp.bert_modeling import BertConfig
from official.nlp.bert_models import _get_transformer_encoder as get_transformer_encoder

FIELD_ANSWER_PREDICTION = 'answer_prediction'


class VCRBert(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VCRBert, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VCRBert):
      raise ValueError('Options has to be an VCRBert proto.')

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    (answer_choices, answer_choices_len,
     answer_label) = (inputs[InputFields.answer_choices_with_question],
                      inputs[InputFields.answer_choices_with_question_len],
                      inputs[InputFields.answer_label])

    # Create model layers.
    token_to_id_layer = token_to_id.TokenToIdLayer(options.bert_vocab_file,
                                                   options.bert_unk_token_id)

    bert_config = BertConfig.from_json_file(options.bert_config_file)
    self.transformer_encoder = get_transformer_encoder(bert_config, None)

    checkpoint = tf.train.Checkpoint(model=self.transformer_encoder)
    self.transformer_encoder_load_status = checkpoint.restore(
        options.bert_checkpoint_file)

    var = self.transformer_encoder.trainable_variables[-1]
    tf.compat.v1.summary.histogram('biases/' + var.op.name, var)

    logit_layer = tf.keras.layers.Dense(1, activation=None)

    # Convert tokens into token ids.
    batch_size = answer_choices.shape[0]
    answer_choices_token_ids = token_to_id_layer(answer_choices)

    # Answer choices' next_sentence prediction feature.
    answer_choices_mask = tf.sequence_mask(answer_choices_len,
                                           maxlen=tf.shape(answer_choices)[-1])
    answer_choices_mask_reshaped = tf.reshape(answer_choices_mask,
                                              [batch_size * NUM_CHOICES, -1])
    answer_choices_token_ids_reshaped = tf.reshape(
        answer_choices_token_ids, [batch_size * NUM_CHOICES, -1])
    _, answer_choices_cls_feature_reshaped = self.transformer_encoder(
        [
            answer_choices_token_ids_reshaped, answer_choices_mask_reshaped,
            tf.zeros_like(answer_choices_token_ids_reshaped, dtype=tf.int32)
        ],
        training=is_training)

    answer_choices_cls_feature = tf.reshape(answer_choices_cls_feature_reshaped,
                                            [batch_size, NUM_CHOICES, -1])

    output = logit_layer(answer_choices_cls_feature)
    output = tf.squeeze(output, axis=-1)

    return {FIELD_ANSWER_PREDICTION: output}

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=inputs[InputFields.answer_label],
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

  def get_scaffold(self):
    """Returns a scaffold object used to initialize variables.

    Returns:
      A tf.train.Scaffold instance.
    """

    def _init_fn(scaffold, sess):
      self.transformer_encoder_load_status.initialize_or_restore(sess)

    scaffold = tf.compat.v1.train.Scaffold(init_fn=_init_fn)
    return scaffold

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    options = self._model_proto

    bert_frozen_variables = []
    if not options.bert_finetune_all:
      bert_frozen_variables = self.transformer_encoder.trainable_variables
      for layer_name in options.bert_finetune_layers:
        bert_frozen_variables = [
            x for x in bert_frozen_variables if layer_name not in x.op.name
        ]

    var_list = [
        x for x in tf.compat.v1.trainable_variables()
        if not x in bert_frozen_variables
    ]
    return var_list
