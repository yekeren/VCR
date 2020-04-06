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
from modeling.models import rnn
from vcr.model_base import ModelBase

from readers.vcr_text_only_reader import InputFields
from readers.vcr_text_only_reader import NUM_CHOICES
import tf_slim as slim

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


def create_embedding_matrix(glove_file, vocab_file, init_width=0.03):
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


class VCRBiLSTMNextSentence(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VCRBiLSTMNextSentence, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VCRBiLSTMNextSentence):
      raise ValueError('Options has to be an VCRBiLSTMNextSentence proto.')

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

    (answer_choices, answer_choices_len) = (
        inputs[InputFields.answer_choices_with_question],
        inputs[InputFields.answer_choices_with_question_len])
    batch_size = answer_choices.shape[0]

    # Convert tokens into token ids.
    token_to_id_layer = token_to_id.TokenToIdLayer(options.vocab_file,
                                                   options.unk_token_id)
    answer_choices_token_ids = token_to_id_layer(answer_choices)
    answer_choices_token_ids = tf.reshape(answer_choices_token_ids,
                                          [batch_size * NUM_CHOICES, -1])

    answer_choices_mask = tf.sequence_mask(answer_choices_len,
                                           maxlen=tf.shape(answer_choices)[-1])
    answer_choices_mask = tf.reshape(answer_choices_mask,
                                     [batch_size * NUM_CHOICES, -1])
    # Convert word ids to embedding vectors.
    glove_embedding_array = create_embedding_matrix(options.glove_file,
                                                    options.vocab_file)
    embedding = tf.get_variable('word/embedding',
                                initializer=glove_embedding_array,
                                trainable=True)
    answer_choices_embs = tf.nn.embedding_lookup(embedding,
                                                 answer_choices_token_ids,
                                                 max_norm=None)

    # Encode the sequence using BiLSTM model.
    with tf.variable_scope('answer_choice_encoder'):
      _, final_features = rnn.RNN(answer_choices_embs,
                                  tf.reshape(answer_choices_len,
                                             [batch_size * NUM_CHOICES]),
                                  options.rnn_config,
                                  is_training=is_training)

    # MLP.
    with slim.arg_scope(fc_scope_fn()):
      with tf.variable_scope('classification'):
        with tf.variable_scope('hidden'):
          output = tf.contrib.layers.fully_connected(final_features,
                                                     num_outputs=1024,
                                                     activation_fn=tf.nn.relu)
          output = tf.contrib.layers.dropout(
              output,
              keep_prob=options.dropout_keep_prob,
              is_training=is_training)
        with tf.variable_scope('output'):
          output = tf.contrib.layers.fully_connected(output,
                                                     num_outputs=1,
                                                     activation_fn=None)
          output = tf.reshape(output, [batch_size, NUM_CHOICES])

    return {
        FIELD_ANSWER_PREDICTION: output,
    }

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    # losses = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=tf.one_hot(inputs[InputFields.answer_label], depth=NUM_CHOICES),
    #     logits=predictions[FIELD_ANSWER_PREDICTION])
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
