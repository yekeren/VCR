from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
import tf_slim as slim

from object_detection.builders.model_builder import _build_faster_rcnn_feature_extractor as build_faster_rcnn_feature_extractor

from protos import fast_rcnn_pb2


class FastRCNNLayer(tf.keras.layers.Layer):
  """FastRCNN layer."""

  def __init__(self, options, **kwargs):
    """Initializes the FastRCNN feature extractor.

    Args:
      options: A fast_rcnn_pb2.FastRCNN proto.
    """
    super(FastRCNNLayer, self).__init__(**kwargs)

    if not isinstance(options, fast_rcnn_pb2.FastRCNN):
      raise ValueError('The options has to be a fast_rcnn_pb2.FastRCNN proto!')

    self.options = options

  def call(self, inputs, proposals, training=True):
    """Converts the inputs to token_ids.

    Args:
      inputs: Input image, a [batch, height, width, 3] float tensor. The pixel
        values are in the range of [0, 255].
      proposals: Boxes used to crop the image features, using normalized
        coordinates. It should be a [batch, max_num_proposals, 4] float tensor
        denoting [y1, x1, y2, x2].
      training: If true, the layer is executed in training mode.

    Returns:
      A tf.int32 tensor which has the same shape as the inputs.
    """
    if tf.executing_eagerly():
      raise ValueError('`FastRCNNLayer` does not support eager execution!')

    is_training = training

    # Create the feature extractor based on the config proto.
    options = self.options
    feature_extractor = build_faster_rcnn_feature_extractor(
        feature_extractor_config=options.feature_extractor,
        inplace_batchnorm_update=options.inplace_batchnorm_update,
        is_training=is_training)

    # Extract `features_to_crop` from the original image.
    #   shape = [batch, feature_map_h, feature_map_w, feature_map_d].
    preprocessed_inputs = feature_extractor.preprocess(inputs)
    features_to_crop, _ = feature_extractor.extract_proposal_features(
        preprocessed_inputs, scope='FirstStageFeatureExtractor')

    if options.dropout_on_feature_map:
      features_to_crop = slim.dropout(features_to_crop,
                                      keep_prob=options.dropout_keep_prob,
                                      is_training=is_training)

    # Crop `flattened_proposal_features_maps`.
    #   shape = [batch*max_num_proposals, crop_size, crop_size, feature_depth].
    batch = proposals.shape[0]
    max_num_proposals = tf.shape(proposals)[1]

    box_ind = tf.expand_dims(tf.range(batch), axis=-1)
    box_ind = tf.tile(box_ind, [1, max_num_proposals])

    cropped_regions = tf.image.crop_and_resize(
        features_to_crop,
        boxes=tf.reshape(proposals, [-1, 4]),
        box_ind=tf.reshape(box_ind, [-1]),
        crop_size=[options.initial_crop_size, options.initial_crop_size])

    flattened_proposal_features_maps = slim.max_pool2d(
        cropped_regions,
        [options.maxpool_kernel_size, options.maxpool_kernel_size],
        stride=options.maxpool_stride)

    # Extract `proposal_features`,
    #   shape = [batch, max_num_proposals, feature_dims].
    box_classifier_features = feature_extractor.extract_box_classifier_features(
        flattened_proposal_features_maps, scope='SecondStageFeatureExtractor')

    flattened_roi_pooled_features = tf.reduce_mean(box_classifier_features,
                                                   [1, 2],
                                                   name='AvgPool')
    flattened_roi_pooled_features = slim.dropout(
        flattened_roi_pooled_features,
        keep_prob=options.dropout_keep_prob,
        is_training=is_training)

    proposal_features = tf.reshape(flattened_roi_pooled_features,
                                   [batch, max_num_proposals, -1])

    # Assign weights from pre-trained checkpoint.
    tf.compat.v1.train.init_from_checkpoint(options.checkpoint_path,
                                            assignment_map={"/": "/"})
    return proposal_features
