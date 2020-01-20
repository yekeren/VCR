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
from modeling.models.model_base import ModelBase
from modeling.utils import visualization
from modeling.utils import masked_ops

from readers.vcr_reader import InputFields
from readers.vcr_reader import NUM_CHOICES

from bert.modeling import BertConfig
from bert.modeling import BertModel
from bert.modeling import get_assignment_map_from_checkpoint

FIELD_ANSWER_PREDICTION = 'answer_prediction'


def _to_batch_coordinates(bboxes, height, width, max_height, max_width):
  """Converts the normalized coordinates to be relative to the batch images.

  Args:
    bboxes: Normalized coordinates, a [..., 4] float tensor (values are in the
      range of [0, 1]) denoting [ymin, xmin, ymax, xmax].
    height: Image height.
    width: Image width.
    max_height: Maximum image height in the batch.
    max_width: Maximum image width in the batch.

  Returns:
    Normalized coordinates relative to the batch images.
  """
  height, width, max_height, max_width = (tf.cast(height, tf.float32),
                                          tf.cast(width, tf.float32),
                                          tf.cast(max_height, tf.float32),
                                          tf.cast(max_width, tf.float32))
  height, width = tf.expand_dims(height, axis=1), tf.expand_dims(width, axis=1)

  ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=-1)
  return tf.stack([
      ymin * height / max_height, xmin * width / max_width,
      ymax * height / max_height, xmax * width / max_width
  ], -1)


class VCRVBert(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VCRVBert, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VCRVBert):
      raise ValueError('Options has to be an VCRVBert proto.')

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    (image, height, width, num_objects, object_bboxes, object_labels,
     object_scores, answer_choices, answer_choices_len,
     answer_label) = (inputs[InputFields.img_data],
                      inputs[InputFields.img_height],
                      inputs[InputFields.img_width],
                      inputs[InputFields.num_objects],
                      inputs[InputFields.object_bboxes],
                      inputs[InputFields.object_labels],
                      inputs[InputFields.object_scores],
                      inputs[InputFields.answer_choices_with_question],
                      inputs[InputFields.answer_choices_with_question_len],
                      inputs[InputFields.answer_label])

    # Visualize image and object bboxes.
    batch_size = image.shape[0]

    image_batch_shape = tf.shape(image)
    object_bboxes = _to_batch_coordinates(object_bboxes, height, width,
                                          image_batch_shape[1],
                                          image_batch_shape[2])
    image_with_boxes = visualization.draw_bounding_boxes_on_image_tensors(
        image, num_objects, object_bboxes, object_labels, object_scores)
    tf.summary.image('vcr/detection', image_with_boxes, max_outputs=10)

    # Extract FRCNN feature.
    frcnn_features = fast_rcnn.FastRCNN(tf.cast(image, tf.float32),
                                        object_bboxes,
                                        options=options.fast_rcnn_config,
                                        is_training=is_training)
    object_masks = tf.sequence_mask(num_objects,
                                    tf.shape(object_bboxes)[1],
                                    dtype=tf.float32)
    image_feature = masked_ops.masked_avg_nd(frcnn_features,
                                             object_masks,
                                             dim=1)

    # Convert tokens into token ids.
    token_to_id_layer = token_to_id.TokenToIdLayer(options.bert_vocab_file,
                                                   options.bert_unk_token_id)
    answer_choices_token_ids = token_to_id_layer(answer_choices)
    answer_choices_token_ids_reshaped = tf.reshape(
        answer_choices_token_ids, [batch_size * NUM_CHOICES, -1])

    answer_choices_mask = tf.sequence_mask(answer_choices_len,
                                           maxlen=tf.shape(answer_choices)[-1])
    answer_choices_mask_reshaped = tf.reshape(answer_choices_mask,
                                              [batch_size * NUM_CHOICES, -1])

    # Bert prediction.
    bert_config = BertConfig.from_json_file(options.bert_config_file)
    bert_model = BertModel(bert_config,
                           is_training,
                           input_ids=answer_choices_token_ids_reshaped,
                           input_mask=answer_choices_mask_reshaped)

    answer_choices_cls_feature_reshaped = bert_model.get_pooled_output()
    answer_choices_cls_feature = tf.reshape(answer_choices_cls_feature_reshaped,
                                            [batch_size, NUM_CHOICES, -1])

    assignment_map, _ = get_assignment_map_from_checkpoint(
        tf.global_variables(), options.bert_checkpoint_file)

    # Fuse image feature.
    image_feature_tiled = tf.tile(image_feature, [1, NUM_CHOICES, 1])
    answer_choices_cls_feature = tf.concat(
        [answer_choices_cls_feature, image_feature_tiled], -1)

    # Classification layer.
    output = tf.compat.v1.layers.dense(answer_choices_cls_feature,
                                       units=1,
                                       activation=None)
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

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    options = self._model_proto

    bert_frozen_variables = []
    if not options.bert_finetune_all:
      bert_frozen_variables = [
          x for x in tf.compat.v1.trainable_variables() if 'bert' in x.op.name
      ]
      for layer_name in options.bert_finetune_layers:
        bert_frozen_variables = [
            x for x in bert_frozen_variables if layer_name not in x.op.name
        ]

    var_list = [
        x for x in tf.compat.v1.trainable_variables()
        if not x in bert_frozen_variables
    ]
    return var_list
