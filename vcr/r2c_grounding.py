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
from vcr.model_base import ModelBase

from readers.vcr_reader import InputFields
from readers.vcr_reader import NUM_CHOICES

FIELD_ANSWER_PREDICTION = 'answer_prediction'


def load_embeddings(filename):
  embeddings_index = {}
  with open(filename, 'r', encoding='utf8') as f:
    for i, line in enumerate(f):
      word, coefs = line.split(maxsplit=1)
      coefs = np.fromstring(coefs, dtype=np.float32, sep=' ')
      embeddings_index[word] = coefs
      if (i + 1) % 10000 == 0:
        logging.info('Load embedding %i.', i + 1)
  return embeddings_index


def load_vocabulary(filename):
  with open(filename, 'r', encoding='utf8') as f:
    return [x.strip('\n') for x in f]


def create_embedding_matrix(glove_file, vocab_file, init_width=0.01):
  embeddings_index = load_embeddings(glove_file)
  embedding_dims = embeddings_index['the'].shape[-1]
  vocab = load_vocabulary(vocab_file)

  embedding_matrix = np.random.uniform(-init_width, init_width,
                                       (len(vocab), embedding_dims))
  for i, word in enumerate(vocab):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  return embedding_matrix.astype(np.float32)


class VCRR2CGrounding(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VCRR2CGrounding, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VCRR2CGrounding):
      raise ValueError('Options has to be an VCRR2CGrounding proto.')

  def _get_token_embedding_vectors(self, inputs, options):
    """Gets token embedding vectors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      options: A model_pb2.Embedding proto.

    Returns:
      question_embs: A [batch, max_question_len, dims] tensor.
      answer_embs: A [batch, NUM_CHOICES, max_answer_len, dims] tensor.
    """
    if not isinstance(options, model_pb2.Embedding):
      raise ValueError('Options has to be a model_pb2.Embedding proto.')

    oneof = options.WhichOneof('embedding_oneof')

    # Returns the pre-extracted Bert embeddings.
    if oneof == 'bert_embedding':
      return (inputs[InputFields.question_bert],
              inputs[InputFields.answer_choices_bert])

    # Extract GloVe embeddings.
    options = options.glove_embedding
    token_to_id_fn = token_to_id.TokenToIdLayer(options.vocab_file,
                                                options.unk_token_id)
    question_token_ids = token_to_id_fn(inputs[InputFields.question])
    answer_token_ids = token_to_id_fn(inputs[InputFields.answer_choices])

    glove_embedding_array = create_embedding_matrix(options.glove_file,
                                                    options.vocab_file)
    with tf.variable_scope('word_embedding'):
      glove_embedding = tf.get_variable('weights',
                                        initializer=glove_embedding_array,
                                        trainable=True)
    question_embs = tf.nn.embedding_lookup(glove_embedding, question_token_ids)
    answer_embs = tf.nn.embedding_lookup(glove_embedding, answer_token_ids)
    return question_embs, answer_embs

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    (num_objects, object_bboxes, object_labels, object_scores, object_features,
     question_tags, question_len, answer_tags, answer_len,
     answer_label) = (inputs[InputFields.num_objects],
                      inputs[InputFields.object_bboxes],
                      inputs[InputFields.object_labels],
                      inputs[InputFields.object_scores],
                      inputs[InputFields.object_features],
                      inputs[InputFields.question_tag],
                      inputs[InputFields.question_len],
                      inputs[InputFields.answer_choices_tag],
                      inputs[InputFields.answer_choices_len],
                      inputs[InputFields.answer_label])

    batch_size, object_feature_dims = (object_features.shape[0],
                                       object_features.shape[2])
    object_features = tf.concat(
        [object_features,
         tf.zeros([batch_size, 1, object_feature_dims])],
        axis=1)

    # Get the word embeddings.
    question_embs, answer_embs = self._get_token_embedding_vectors(
        inputs, options.embedding_config)
    embedding_dims = question_embs.shape[-1]

    # Reshape answer-related tensors
    # to the shape of [batch_size * NUM_CHOICES, max_seq_len, ...].
    answer_embs = tf.reshape(answer_embs,
                             [batch_size * NUM_CHOICES, -1, embedding_dims])
    answer_tags = tf.reshape(answer_tags, [batch_size * NUM_CHOICES, -1])
    answer_len = tf.reshape(answer_len, [-1])

    # Ground both theq question and the answer choices.
    question_object_feature_indices = tf.stack([
        tf.tile(tf.expand_dims(tf.range(batch_size), axis=1),
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
    with tf.variable_scope('question_encoder'):
      _, question_features = rnn.RNN(tf.concat(
          [question_embs, question_object_features], -1),
                                     question_len,
                                     options.rnn_config,
                                     is_training=is_training)

    with tf.variable_scope('answer_choice_encoder'):
      _, answer_features = rnn.RNN(tf.concat(
          [answer_embs, answer_object_features], -1),
                                   answer_len,
                                   options.rnn_config,
                                   is_training=is_training)

    # Compute the joint representation.
    question_features = tf.gather(tf.expand_dims(question_features, 1),
                                  [0] * NUM_CHOICES,
                                  axis=1)
    question_features = tf.reshape(question_features,
                                   [batch_size * NUM_CHOICES, -1])
    inputs = tf.concat([
        question_features, answer_features, question_features * answer_features
    ], -1)
    output = tf.compat.v1.layers.dense(inputs, units=1, activation=None)
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
