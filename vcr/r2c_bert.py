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
from vcr.model_base import ModelBase

from readers.vcr_reader import InputFields
from readers.vcr_reader import NUM_CHOICES
import tf_slim as slim

FIELD_ANSWER_PREDICTION = 'answer_prediction'


class R2CBert(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(R2CBert, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.R2CBert):
      raise ValueError('Options has to be an R2CBert proto.')

  def _get_class_embedding_vectors(self,
                                   label,
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
    token_to_id_fn = token_to_id.TokenToIdLayer(vocab_file,
                                                unk_token_id=vocab_size)
    label_ids = token_to_id_fn(label)
    tf.compat.v1.summary.histogram('attention/object_ids', label_ids)

    with tf.variable_scope('object_embedding'):
      object_embedding = tf.get_variable('weights',
                                         shape=[vocab_size + 1, embedding_dims],
                                         trainable=True)
    label_embedding = tf.nn.embedding_lookup(object_embedding, label_ids)

    return label_embedding

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """

    with slim.arg_scope(
        hyperparams.build_hyperparams(self._model_proto.hyperparams,
                                      self._is_training)):
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

    (num_objects, object_bboxes, object_labels, object_scores, object_features,
     question_toks, question_embs, question_tags, question_len, answer_toks,
     answer_embs, answer_tags, answer_len, answer_label) = (
         inputs[InputFields.num_objects], inputs[InputFields.object_bboxes],
         inputs[InputFields.object_labels], inputs[InputFields.object_scores],
         inputs[InputFields.object_features], inputs[InputFields.question],
         inputs[InputFields.question_bert], inputs[InputFields.question_tag],
         inputs[InputFields.question_len], inputs[InputFields.answer_choices],
         inputs[InputFields.answer_choices_bert],
         inputs[InputFields.answer_choices_tag],
         inputs[InputFields.answer_choices_len],
         inputs[InputFields.answer_label])

    batch_size, object_feature_dims = (object_features.shape[0],
                                       object_features.shape[2])

    # Merge class embedding and project.
    object_embeddings = self._get_class_embedding_vectors(
        object_labels, options.label_file, options.label_vocab_size,
        options.label_embedding_dims)
    tf.compat.v1.summary.histogram('attention/object_features', object_features)
    tf.compat.v1.summary.histogram('attention/object_embeddings',
                                   object_embeddings)

    object_features = tf.concat([object_features, object_embeddings], -1)
    with tf.variable_scope('object_projection'):
      object_features = tf.contrib.layers.fully_connected(
          object_features,
          num_outputs=options.visual_feature_dims,
          activation_fn=tf.nn.relu)
      object_features = tf.contrib.layers.dropout(object_features,
                                                  keep_prob=0.7,
                                                  is_training=is_training)

    object_features = tf.concat(
        [object_features,
         tf.zeros([batch_size, 1, object_features.shape[-1]])], 1)
    embedding_dims = question_embs.shape[-1]

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

    # Ground both theq question and the answer choices.
    question_object_feature_indices = tf.stack([
        tf.tile(tf.expand_dims(tf.range(batch_size * NUM_CHOICES), axis=1),
                [1, tf.shape(question_tags)[1]]), question_tags
    ], -1)
    question_object_features = tf.gather_nd(object_features,
                                            question_object_feature_indices)
    answer_object_feature_indices = tf.stack([
        tf.tile(tf.expand_dims(tf.range(batch_size * NUM_CHOICES), axis=1),
                [1, tf.shape(answer_tags)[1]]), answer_tags
    ], -1)
    answer_object_features = tf.gather_nd(object_features,
                                          answer_object_feature_indices)

    # Encode the sequence using BiLSTM model.
    question_rnn_inputs, answer_rnn_inputs = question_embs, answer_embs
    if not options.no_vision_representation:
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

    # Attention over the question.
    #   qa_similarity: [batch*NUM_CHOICES, question_len, answer_len].
    #   qa_mask: [batch*NUM_CHOICES, question_len].
    #   qa_attention_weights: [batch*NUM_CHOICES, question_len, answer_len].
    #   question_seq_features: [batch*NUM_CHOICES, question_len, dims].
    #   attended_question: [batch*NUM_CHOICES, answer_len, dims].
    with tf.variable_scope('qa_bilinear'):
      qa_similarity = attention_ops.bilinear(question_seq_features,
                                             answer_seq_features)
      tf.compat.v1.summary.histogram('attention/qa_similarity', qa_similarity)
    qa_mask = tf.expand_dims(
        tf.sequence_mask(question_len,
                         tf.shape(question_seq_features)[1],
                         dtype=tf.float32), 2)
    qa_attention_weights = masked_ops.masked_softmax(data=qa_similarity,
                                                     mask=qa_mask,
                                                     dim=1)
    attended_question = tf.reduce_sum(
        tf.multiply(tf.expand_dims(qa_attention_weights, 3),
                    tf.expand_dims(question_seq_features, 2)), 1)
    # attended_question = tf.einsum('bqa,bqd->bad', qa_attention_weights,
    #                               question_seq_features)

    # Attention over the objects.
    #   oa_similarity: [batch*NUM_CHOICES, object_len, answer_len]
    #   oa_mask: [batch*NUM_CHOICES, object_len].
    #   oa_attention_weights: [batch*NUM_CHOICES, object_len, answer_len].
    #   attended_objects: [batch*NUM_CHOICES, answer_len, dims].

    object_features = tf.tile(object_features, [NUM_CHOICES, 1, 1])
    num_objects = tf.tile(num_objects, [NUM_CHOICES])

    with tf.variable_scope('oa_bilinear'):
      oa_similarity = attention_ops.bilinear(object_features,
                                             answer_seq_features)
      tf.compat.v1.summary.histogram('attention/oa_similarity', oa_similarity)
    oa_mask = tf.expand_dims(
        tf.sequence_mask(num_objects,
                         tf.shape(object_features)[1],
                         dtype=tf.float32), 2)
    oa_attention_weights = masked_ops.masked_softmax(data=oa_similarity,
                                                     mask=oa_mask,
                                                     dim=1)
    attended_objects = tf.reduce_sum(
        tf.multiply(tf.expand_dims(oa_attention_weights, 3),
                    tf.expand_dims(object_features, 2)), 1)

    # Reasoning.
    reasoning_rnn_input = tf.concat(
        [answer_seq_features, attended_question, attended_objects], -1)

    with tf.variable_scope('reasoning'):
      reasoning_seq_features, _ = rnn.RNN(reasoning_rnn_input,
                                          answer_len,
                                          options.rnn_config,
                                          is_training=is_training)

    # Pool features from the sequence.
    final_seq_features = tf.concat(
        [reasoning_seq_features, answer_seq_features, attended_question], -1)

    final_features = masked_ops.masked_avg_nd(
        data=final_seq_features,
        mask=tf.sequence_mask(answer_len,
                              tf.shape(final_seq_features)[1],
                              dtype=tf.float32),
        dim=1)
    final_features = tf.squeeze(final_features, 1)

    # Compute the joint representation.
    with tf.variable_scope('final_mlp'):
      with tf.variable_scope('layer1'):
        output = tf.contrib.layers.fully_connected(final_features,
                                                   num_outputs=1024,
                                                   activation_fn=tf.nn.relu)
      output = tf.contrib.layers.dropout(output,
                                         keep_prob=0.7,
                                         is_training=is_training)
      with tf.variable_scope('layer2'):
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
    return tf.compat.v1.trainable_variables()
