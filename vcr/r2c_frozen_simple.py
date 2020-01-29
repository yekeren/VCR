from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json
import numpy as np
import tensorflow as tf

from protos import model_pb2
from modeling.layers import token_to_id
from modeling.models import rnn
from modeling.utils import masked_ops
from modeling.utils import attention_ops
from modeling.utils import hyperparams
from modeling.utils import visualization
from modeling.models import fast_rcnn
from vcr.model_base import ModelBase

from readers.vcr_reader import InputFields
from readers.vcr_reader import NUM_CHOICES
import tf_slim as slim

FIELD_ANSWER_PREDICTION = 'answer_prediction'


def _convert_to_batch_coords(bboxes, height, width, max_height, max_width):
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


def _trim_to_max_num_objects(num_objects,
                             object_bboxes,
                             object_labels,
                             object_scores,
                             object_features,
                             max_num_objects=10):
  """Trims to the `max_num_objects` objects.

  Args:
    num_objects: A [batch] int tensor.
    object_bboxes: A [batch, pad_num_objects, 4] float tensor.
    object_labels: A [batch, pad_num_objects] int tensor.
    object_scores: A [batch, pad_num_objects] float tensor.
    object_features: A [batch, pad_num_objects, feature_dims] float tensor.
    max_num_objects: Maximum number of objects.

  Returns:
    num_objects: A [batch] int tensor.
    object_bboxes: A [batch, max_num_objects, 4] float tensor.
    object_labels: A [batch, max_num_objects] int tensor.
    object_scores: A [batch, max_num_objects] float tensor.
    object_features: A [batch, max_num_objects, feature_dims] float tensor.
    max_num_objects: Maximum number of objects in the batch.
  """
  max_num_objects = tf.minimum(tf.reduce_max(num_objects), max_num_objects)
  num_objects = tf.minimum(max_num_objects, num_objects)
  object_bboxes = object_bboxes[:, :max_num_objects, :]
  object_labels = object_labels[:, :max_num_objects]
  object_scores = object_scores[:, :max_num_objects]
  object_features = object_features[:, :max_num_objects, :]
  return (num_objects, object_bboxes, object_labels, object_scores,
          object_features, max_num_objects)


def _assign_invalid_tags(tags, max_num_objects):
  """Assigns `-1` to invalid tags.

  Args:
    tags: A int tensor, each value is a index in the objects array.
    max_num_objects: Maximum number of objects in the batch.
  """
  return tf.where(tags >= max_num_objects, -tf.ones_like(tags, tf.int32), tags)


def _get_class_embedding_vectors(label,
                                 vocab_file,
                                 vocab_size,
                                 embedding_dims=300):
  """Gets token embedding vectors.

  Args:
    label: A string tensor.
    vocab_file: Path to the vocabulary file.
    vocab_size: Size of the vocabulary.
    embedding_dims: Dimensions of the embedding vectors.

  Returns:
    label_embedding: Embedding of the label.
  """
  label_ids = token_to_id.TokenToIdLayer(vocab_file,
                                         unk_token_id=vocab_size)(label)
  with tf.variable_scope('object_embedding'):
    object_embedding = tf.get_variable('weights',
                                       shape=[vocab_size + 1, embedding_dims],
                                       trainable=True)
  return tf.nn.embedding_lookup(object_embedding, label_ids)


def _project_object_features(object_features,
                             object_embeddings,
                             output_dims,
                             dropout_keep_prob=1.0,
                             is_training=False):
  """Projects object features to `visual_feature_dims` dimensions.

  Args:
    object_features: A [batch, max_num_objects, feature_dims] float tensor.
    object_embeddings: A [batch, max_num_objects, embedding_dims] float tensor.
  """
  object_features = tf.concat([object_features, object_embeddings], -1)

  with tf.variable_scope('object_projection'):
    object_features = tf.contrib.layers.fully_connected(
        object_features, num_outputs=output_dims, activation_fn=tf.nn.relu)
    object_features = tf.contrib.layers.dropout(object_features,
                                                keep_prob=dropout_keep_prob,
                                                is_training=is_training)

  # Add ZEROs to the end.
  batch_size, _, object_feature_dims = object_features.shape
  object_features = tf.concat(
      [object_features,
       tf.zeros([batch_size, 1, object_feature_dims])], 1)

  return object_features


class VCRR2CFrozenSimple(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VCRR2CFrozenSimple, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VCRR2CFrozenSimple):
      raise ValueError('Options has to be an VCRR2CFrozenSimple proto.')

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    fc_scope_fn = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                is_training)

    with slim.arg_scope(fc_scope_fn()):
      return self._predict(inputs, **kwargs)

  def _predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    (question_toks, question_embs, question_len, answer_toks, answer_embs,
     answer_len, answer_label) = (inputs[InputFields.question],
                                  inputs[InputFields.question_bert],
                                  inputs[InputFields.question_len],
                                  inputs[InputFields.answer_choices],
                                  inputs[InputFields.answer_choices_bert],
                                  inputs[InputFields.answer_choices_len],
                                  inputs[InputFields.answer_label])
    embedding_dims = question_embs.shape[-1]

    # Trim to `max_num_objects` objects.
    (num_objects, object_bboxes, object_labels, object_scores, object_features,
     max_num_objects) = _trim_to_max_num_objects(
         inputs[InputFields.num_objects],
         inputs[InputFields.object_bboxes],
         inputs[InputFields.object_labels],
         inputs[InputFields.object_scores],
         inputs[InputFields.object_features],
         max_num_objects=options.max_num_objects)
    batch_size = object_features.shape[0]

    question_tags = _assign_invalid_tags(inputs[InputFields.question_tag],
                                         max_num_objects)
    answer_tags = _assign_invalid_tags(inputs[InputFields.answer_choices_tag],
                                       max_num_objects)

    # Merge class label embeddings to the Fast-RCNN features.
    object_embeddings = _get_class_embedding_vectors(
        object_labels, options.label_file, options.label_vocab_size,
        options.label_embedding_dims)

    object_features = _project_object_features(
        object_features,
        object_embeddings,
        output_dims=options.visual_feature_dims,
        dropout_keep_prob=options.dropout_keep_prob,
        is_training=is_training)

    # Reshape answer-related tensors
    # to the shape of [batch_size * NUM_CHOICES, max_seq_len, ...].
    question_embs = tf.reshape(question_embs,
                               [batch_size * NUM_CHOICES, -1, embedding_dims])
    question_tags = tf.tile(question_tags, [NUM_CHOICES, 1])
    question_len = tf.tile(question_len, [NUM_CHOICES])

    answer_embs = tf.reshape(answer_embs,
                             [batch_size * NUM_CHOICES, -1, embedding_dims])
    answer_tags = tf.reshape(answer_tags, [batch_size * NUM_CHOICES, -1])
    answer_len = tf.reshape(answer_len, [-1])

    # Ground both the question and the answer choices.
    object_features_tiled = tf.reshape(
        tf.gather(tf.expand_dims(object_features, 1), [0] * NUM_CHOICES,
                  axis=1),
        [batch_size * NUM_CHOICES, -1, object_features.shape[-1]])

    question_object_feature_indices = tf.stack([
        tf.tile(tf.expand_dims(tf.range(batch_size * NUM_CHOICES), axis=1),
                [1, tf.shape(question_tags)[1]]), question_tags
    ], -1)
    question_object_features = tf.gather_nd(object_features_tiled,
                                            question_object_feature_indices)
    answer_object_feature_indices = tf.stack([
        tf.tile(tf.expand_dims(tf.range(batch_size * NUM_CHOICES), axis=1),
                [1, tf.shape(answer_tags)[1]]), answer_tags
    ], -1)
    answer_object_features = tf.gather_nd(object_features_tiled,
                                          answer_object_feature_indices)

    # Encode the sequence using BiLSTM model.
    question_rnn_inputs, answer_rnn_inputs = question_embs, answer_embs
    question_rnn_inputs = tf.concat([question_embs, question_object_features],
                                    -1)
    answer_rnn_inputs = tf.concat([answer_embs, answer_object_features], -1)

    with tf.variable_scope('grounding_encoder'):
      question_seq_features, _ = rnn.RNN(question_rnn_inputs,
                                         question_len,
                                         options.rnn_config,
                                         is_training=is_training)

    with tf.variable_scope('grounding_encoder', reuse=True):
      answer_seq_features, _ = rnn.RNN(answer_rnn_inputs,
                                       answer_len,
                                       options.rnn_config,
                                       is_training=is_training)

    ## # Attention over the question.
    ## #   qa_similarity: [batch*NUM_CHOICES, question_len, answer_len].
    ## #   qa_mask: [batch*NUM_CHOICES, question_len].
    ## #   qa_attention_weights: [batch*NUM_CHOICES, question_len, answer_len].
    ## #   question_seq_features: [batch*NUM_CHOICES, question_len, dims].
    ## #   attended_question: [batch*NUM_CHOICES, answer_len, dims].
    ## with tf.variable_scope('qa_bilinear'):
    ##   qa_similarity = attention_ops.bilinear(question_seq_features,
    ##                                          answer_seq_features)
    ##   tf.compat.v1.summary.histogram('attention/qa_similarity', qa_similarity)
    ## qa_mask = tf.expand_dims(
    ##     tf.sequence_mask(question_len,
    ##                      tf.shape(question_seq_features)[1],
    ##                      dtype=tf.float32), 2)
    ## qa_attention_weights = masked_ops.masked_softmax(data=qa_similarity,
    ##                                                  mask=qa_mask,
    ##                                                  dim=1)
    ## # attended_question = tf.reduce_sum(
    ## #     tf.multiply(tf.expand_dims(qa_attention_weights, 3),
    ## #                 tf.expand_dims(question_seq_features, 2)), 1)
    ## attended_question = tf.einsum('bqa,bqd->bad', qa_attention_weights,
    ##                               question_seq_features)

    ## # Attention over the objects.
    ## #   oa_similarity: [batch*NUM_CHOICES, object_len, answer_len]
    ## #   oa_mask: [batch*NUM_CHOICES, object_len].
    ## #   oa_attention_weights: [batch*NUM_CHOICES, object_len, answer_len].
    ## #   attended_objects: [batch*NUM_CHOICES, answer_len, dims].

    ## object_features = tf.gather(tf.expand_dims(object_features, 1),
    ##                             [0] * NUM_CHOICES,
    ##                             axis=1)
    ## object_features = tf.reshape(
    ##     object_features,
    ##     [batch_size * NUM_CHOICES, -1, object_features.shape[-1]])
    ## num_objects = tf.tile(num_objects, [NUM_CHOICES])

    ## with tf.variable_scope('oa_bilinear'):
    ##   oa_similarity = attention_ops.bilinear(object_features,
    ##                                          answer_seq_features)
    ##   tf.compat.v1.summary.histogram('attention/oa_similarity', oa_similarity)
    ## oa_mask = tf.expand_dims(
    ##     tf.sequence_mask(num_objects,
    ##                      tf.shape(object_features)[1],
    ##                      dtype=tf.float32), 2)
    ## oa_attention_weights = masked_ops.masked_softmax(data=oa_similarity,
    ##                                                  mask=oa_mask,
    ##                                                  dim=1)
    ## # attended_objects = tf.reduce_sum(
    ## #     tf.multiply(tf.expand_dims(oa_attention_weights, 3),
    ## #                 tf.expand_dims(object_features, 2)), 1)
    ## attended_objects = tf.einsum('boa,bod->bad', oa_attention_weights,
    ##                              object_features)
    ##
    ## # Reasoning.
    ## reasoning_rnn_input = tf.concat(
    ##     [answer_seq_features, attended_question, attended_objects], -1)

    ## with tf.variable_scope('reasoning'):
    ##   reasoning_seq_features, _ = rnn.RNN(reasoning_rnn_input,
    ##                                       answer_len,
    ##                                       options.rnn_config,
    ##                                       is_training=is_training)

    # Pool features from the sequence.
    question_pooled_features = masked_ops.masked_avg_nd(
        data=question_seq_features,
        mask=tf.sequence_mask(question_len,
                              tf.shape(question_seq_features)[1],
                              dtype=tf.float32),
        dim=1)
    answer_pooled_features = masked_ops.masked_avg_nd(
        data=answer_seq_features,
        mask=tf.sequence_mask(answer_len,
                              tf.shape(answer_seq_features)[1],
                              dtype=tf.float32),
        dim=1)
    final_features = tf.concat([
        question_pooled_features, answer_pooled_features,
        question_pooled_features * answer_pooled_features
    ], -1)
    final_features = tf.squeeze(final_features, 1)

    # Compute the joint representation.
    with tf.variable_scope('classification'):
      with tf.variable_scope('hidden'):
        output = tf.contrib.layers.fully_connected(final_features,
                                                   num_outputs=1024,
                                                   activation_fn=tf.nn.relu)
      output = tf.contrib.layers.dropout(output,
                                         keep_prob=options.dropout_keep_prob,
                                         is_training=is_training)
      with tf.variable_scope('output'):
        output = tf.contrib.layers.fully_connected(output,
                                                   num_outputs=1,
                                                   activation_fn=None)

    output = tf.reshape(output, [batch_size, NUM_CHOICES])
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
    var_list = tf.compat.v1.trainable_variables()
    var_list = [
        x for x in var_list if not ('FirstStageFeatureExtractor' in x.op.name)
    ]
    return var_list
