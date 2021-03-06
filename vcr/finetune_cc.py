from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json
import numpy as np
import tensorflow as tf

from protos import model_pb2
from modeling.layers import token_to_id
from modeling.models import fast_rcnn
from modeling.utils import hyperparams
from vcr.model_base import ModelBase

from readers.vcr_fields import InputFields
from readers.vcr_fields import NUM_CHOICES

from bert2.modeling import BertConfig
from bert2.modeling import BertModel
from bert2.modeling import get_assignment_map_from_checkpoint

import tf_slim as slim

FIELD_ANSWER_PREDICTION = 'answer_prediction'

UNK = '[UNK]'
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'

IMG = '[unused400]'


def remove_detections(num_detections,
                      detection_scores,
                      detection_classes,
                      detection_boxes,
                      max_num_detections=10):
  """Trims to the `max_num_detections` objects.

  Args:
    num_detections: A [batch] int tensor.
    detection_scores: A [batch, pad_num_detections] float tensor.
    detection_classes: A [batch, pad_num_detections] int tensor.
    detection_boxes: A [batch, pad_num_detections, 4] float tensor.
    max_num_detections: Maximum number of objects.

  Returns:
    num_detections: A [batch] int tensor.
    detection_scores: A [batch, max_num_detections] float tensor.
    detection_classes: A [batch, max_num_detections] int tensor.
    detection_boxes: A [batch, max_num_detections, 4] float tensor.
  """
  max_num_detections = tf.minimum(tf.reduce_max(num_detections),
                                  max_num_detections)

  num_detections = tf.minimum(max_num_detections, num_detections)
  detection_boxes = detection_boxes[:, :max_num_detections, :]
  detection_classes = detection_classes[:, :max_num_detections]
  detection_scores = detection_scores[:, :max_num_detections]
  return (num_detections, detection_scores, detection_classes, detection_boxes)


def convert_to_batch_coordinates(detection_boxes, height, width, batch_height,
                                 batch_width):
  """Converts the coordinates to be relative to the batch images. """
  height = tf.expand_dims(tf.cast(height, tf.float32), -1)
  width = tf.expand_dims(tf.cast(width, tf.float32), -1)
  batch_height = tf.cast(batch_height, tf.float32)
  batch_width = tf.cast(batch_width, tf.float32)

  ymin, xmin, ymax, xmax = tf.unstack(detection_boxes, axis=-1)
  detection_boxes_converted = tf.stack([
      ymin * height / batch_height, xmin * width / batch_width,
      ymax * height / batch_height, xmax * width / batch_width
  ], -1)
  return detection_boxes_converted


def insert_image_box(num_detections, detection_scores, detection_classes,
                     detection_boxes):
  """Insert image-box to the 0-th position. """
  batch_size = num_detections.shape[0]

  num_detections += 1
  detection_scores = tf.concat(
      [tf.fill([batch_size, 1], 1.0), detection_scores], axis=-1)
  detection_classes = tf.concat(
      [tf.fill([batch_size, 1], IMG), detection_classes], axis=-1)
  detection_boxes = tf.concat([
      tf.gather(tf.constant([[[0, 0, 1, 1]]], dtype=tf.float32),
                [0] * batch_size,
                axis=0), detection_boxes
  ], 1)
  return num_detections, detection_scores, detection_classes, detection_boxes


class FinetuneCC(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(FinetuneCC, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.FinetuneCC):
      raise ValueError('Options has to be an FinetuneCC proto.')

    options = model_proto

    self._token_to_id_func = token_to_id.TokenToIdLayer(
        options.bert_vocab_file, options.bert_unk_token_id)
    self._bert_config = BertConfig.from_json_file(options.bert_config_file)

    self._slim_fc_scope = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                        is_training)()

    if options.rationale_model:
      self._field_label = InputFields.rationale_label
      self._field_choices = InputFields.rationale_choices_with_question
      self._field_choices_tag = InputFields.rationale_choices_with_question_tag
      self._field_choices_len = InputFields.rationale_choices_with_question_len
    else:
      self._field_label = InputFields.answer_label
      self._field_choices = InputFields.answer_choices_with_question
      self._field_choices_tag = InputFields.answer_choices_with_question_tag
      self._field_choices_len = InputFields.answer_choices_with_question_len

  def create_bert_input_tensors(self, num_detections, detection_classes,
                                detection_features, caption, caption_len):
    """Predicts the matching score of the given image-text pair.

    Args:
      num_detections: A [batch] int tensor.
      detection_classes: A [batch, max_detections] string tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption: A [batch, max_caption_len] string tensor.
      caption_len: A [batch] int tensor.

    Returns:
      input_ids: A [batch, 1 + max_detections + 1 + max_caption_len + 1] int tensor.
      input_masks: A [batch, 1 + max_detections + 1 + max_caption_len + 1] boolean tensor.
      input_features: A [batch, 1 + max_detections + 1 + max_caption_len + 1, dims] float tensor.
    """
    batch_size = num_detections.shape[0]
    token_to_id_func = self._token_to_id_func

    # Create input masks.
    mask_one = tf.fill([batch_size, 1], True)
    max_caption_len = tf.shape(caption)[1]
    max_detections = tf.shape(detection_features)[1]
    input_masks = tf.concat([
        mask_one,
        tf.sequence_mask(num_detections, maxlen=max_detections), mask_one,
        tf.sequence_mask(caption_len, maxlen=max_caption_len), mask_one
    ], -1)

    # Create input tokens.
    token_cls = tf.fill([batch_size, 1], CLS)
    token_sep = tf.fill([batch_size, 1], SEP)
    detection_classes_masked = tf.fill([batch_size, max_detections], MASK)
    input_tokens = tf.concat(
        [token_cls, detection_classes_masked, token_sep, caption, token_sep],
        axis=-1)
    input_ids = token_to_id_func(input_tokens)

    # Create input features.
    feature_dims = detection_features.shape[-1]
    input_features = tf.concat([
        tf.fill([batch_size, 1, feature_dims], 0.0), detection_features,
        tf.fill([batch_size, 2 + max_caption_len, feature_dims], 0.0)
    ], 1)
    return input_ids, input_masks, input_features

  def image_text_matching(self, num_detections, detection_classes,
                          detection_features, caption, caption_len):
    """Predicts the matching score of the given image-text pair.

    Args:
      num_detections: A [batch] int tensor.
      detection_classes: A [batch, max_detections] string tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption: A [batch, max_caption_len] string tensor.
      caption_len: A [batch] int tensor.

    Returns:
      matching_score: A [batch] float tensor.
    """
    (input_ids, input_masks, input_features) = self.create_bert_input_tensors(
        num_detections, detection_classes, detection_features, caption,
        caption_len)
    bert_model = BertModel(self._bert_config,
                           self._is_training,
                           input_ids=input_ids,
                           input_mask=input_masks,
                           input_features=input_features,
                           scope='bert')
    return bert_model.get_pooled_output()

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    # Decode fields from `inputs`.
    (image, image_height, image_width, num_detections, detection_boxes,
     detection_classes, detection_scores) = (
         inputs[InputFields.img_data],
         inputs[InputFields.img_height],
         inputs[InputFields.img_width],
         inputs[InputFields.num_detections],
         inputs[InputFields.detection_boxes],
         inputs[InputFields.detection_classes],
         inputs[InputFields.detection_scores],
     )
    batch_size = image.shape[0]

    # Remove boxes if there are too many.
    (num_detections, detection_scores, detection_classes,
     detection_boxes) = remove_detections(num_detections,
                                          detection_scores,
                                          detection_classes,
                                          detection_boxes,
                                          max_num_detections=10)

    # Insert image-box to the 0-th position.
    (num_detections, detection_scores, detection_classes,
     detection_boxes) = insert_image_box(num_detections, detection_scores,
                                         detection_classes, detection_boxes)

    # Extract Fast-RCNN features.
    image_batch_shape = tf.shape(image)
    detection_boxes = convert_to_batch_coordinates(detection_boxes,
                                                   image_height, image_width,
                                                   image_batch_shape[1],
                                                   image_batch_shape[2])
    detection_features, _ = fast_rcnn.FastRCNN(image,
                                               detection_boxes,
                                               options=options.fast_rcnn_config,
                                               is_training=is_training)
    with slim.arg_scope(self._slim_fc_scope):
      detection_features = slim.fully_connected(detection_features,
                                                self._bert_config.hidden_size,
                                                activation_fn=None,
                                                scope='detection/project')

    # Create BERT prediction.
    choice_lengths = tf.unstack(inputs[self._field_choices_len], axis=1)
    choice_captions = tf.unstack(inputs[self._field_choices], axis=1)
    choice_tags = tf.unstack(inputs[self._field_choices_tag], axis=1)
    assert (NUM_CHOICES == len(choice_captions) == len(choice_lengths) ==
            len(choice_tags))

    reuse = False
    feature_to_predict_choices = []
    for i in range(NUM_CHOICES):
      caption = choice_captions[i]
      caption_len = choice_lengths[i]
      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        feature_to_predict_choices.append(
            self.image_text_matching(num_detections, detection_classes,
                                     detection_features, caption, caption_len))
      reuse = True

    with slim.arg_scope(self._slim_fc_scope):
      features = tf.stack(feature_to_predict_choices, 1)
      logits = slim.fully_connected(features,
                                    num_outputs=1,
                                    activation_fn=None,
                                    scope='itm/logits')
      logits = tf.squeeze(logits, -1)

    # Restore from BERT checkpoint.
    assignment_map, _ = get_assignment_map_from_checkpoint(
        tf.global_variables(), options.bert_checkpoint_file)
    if 'global_step' in assignment_map:
      assignment_map.pop('global_step')
    tf.train.init_from_checkpoint(options.bert_checkpoint_file, assignment_map)

    return {FIELD_ANSWER_PREDICTION: logits}

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

    labels = tf.one_hot(inputs[self._field_label], NUM_CHOICES)
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
    y_true = inputs[self._field_label]
    y_pred = tf.argmax(predictions[FIELD_ANSWER_PREDICTION], -1)

    accuracy_metric.update_state(y_true, y_pred)
    return {'metrics/accuracy': accuracy_metric}

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    options = self._model_proto

    trainable_variables = tf.trainable_variables()
    trainable_variables = [
        x for x in trainable_variables
        if 'FirstStageFeatureExtractor' not in x.op.name
    ]
    return trainable_variables

    # trainable_variables = tf.compat.v1.trainable_variables()

    # # Look for BERT frozen variables.
    # frozen_variables = []
    # for var in trainable_variables:
    #   for name_pattern in options.frozen_variable_patterns:
    #     if name_pattern in var.op.name:
    #       frozen_variables.append(var)
    #       break

    # # Get trainable variables.
    # var_list = list(set(trainable_variables) - set(frozen_variables))
    # return var_list
