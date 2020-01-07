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

FIELD_ANSWER_PREDICTION = 'answer_prediction'


def load_embeddings(filename):
  embeddings_index = {}
  with open(filename, 'r', encoding='utf8') as f:
    for i, line in enumerate(f):
      word, coefs = line.split(maxsplit=1)
      coefs = np.fromstring(coefs, 'f', sep=' ')
      embeddings_index[word] = coefs
      if (i + 1) % 10000 == 0:
        logging.info('Load embedding %i.', i + 1)
  return embeddings_index


def load_vocabulary(filename):
  with open(filename, 'r', encoding='utf8') as f:
    return [x.strip('\n') for x in f]


def create_embedding_matrix(glove_file,
                            vocab_file,
                            embedding_dims,
                            init_width=0.03):
  embeddings_index = load_embeddings(glove_file)
  vocab = load_vocabulary(vocab_file)

  embedding_matrix = np.random.uniform(-init_width, init_width,
                                       (len(vocab), embedding_dims))
  for i, word in enumerate(vocab):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  return embedding_matrix


class VCRBiLSTMConcat(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VCRBiLSTMConcat, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VCRBiLSTMConcat):
      raise ValueError('Options has to an VCRBiLSTMConcat proto.')

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    (question, question_len, answer_choices, answer_choices_len,
     answer_label) = (inputs[InputFields.question],
                      inputs[InputFields.question_len],
                      inputs[InputFields.answer_choices],
                      inputs[InputFields.answer_choices_len],
                      inputs[InputFields.answer_label])

    # Create model layers.
    token_to_id_layer = token_to_id.TokenToIdLayer(options.vocab_file,
                                                   options.unk_token_id)
    embeddings_initializer = 'uniform'
    if options.glove_file:
      embeddings_initializer = tf.keras.initializers.Constant(
          create_embedding_matrix(options.glove_file, options.vocab_file,
                                  options.embedding_dims))
    embedding_layer = tf.keras.layers.Embedding(
        options.vocab_size,
        options.embedding_dims,
        embeddings_initializer=embeddings_initializer)
    question_lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(options.lstm_units,
                             dropout=options.lstm_dropout,
                             recurrent_dropout=options.lstm_recurrent_dropout,
                             return_state=True),
        name='question_bidirectional')
    answer_choice_lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(options.lstm_units,
                             dropout=options.lstm_dropout,
                             recurrent_dropout=options.lstm_recurrent_dropout),
        name='answer_bidirectional')

    # Convert tokens into embeddings.
    batch_size = answer_choices.shape[0]
    (question_token_ids,
     answer_choices_token_ids) = (token_to_id_layer(question),
                                  token_to_id_layer(answer_choices))
    (question_embs,
     answer_choices_embs) = (embedding_layer(question_token_ids),
                             embedding_layer(answer_choices_token_ids))

    # Question LSTM encoder.
    question_mask = tf.sequence_mask(question_len,
                                     maxlen=tf.shape(question)[-1])
    question_outputs = question_lstm_layer(question_embs,
                                           mask=question_mask,
                                           training=is_training)
    question_feature, question_states = (question_outputs[0],
                                         question_outputs[1:])
    question_states_tiled = []
    for question_state in question_states:
      question_state = tf.gather(tf.expand_dims(question_state, axis=1),
                                 indices=[0] * NUM_CHOICES,
                                 axis=1)
      question_states_tiled.append(
          tf.reshape(question_state, [-1, question_state.shape[-1]]))

    # Answer LSTM encoder.
    answer_choices_mask = tf.sequence_mask(answer_choices_len,
                                           maxlen=tf.shape(answer_choices)[-1])
    answer_choices_mask_reshaped = tf.reshape(answer_choices_mask,
                                              [batch_size * NUM_CHOICES, -1])
    answer_choices_embs_reshaped = tf.reshape(
        answer_choices_embs,
        [batch_size * NUM_CHOICES, -1, options.embedding_dims])

    answer_choices_feature_reshaped = answer_choice_lstm_layer(
        answer_choices_embs_reshaped,
        mask=answer_choices_mask_reshaped,
        training=is_training,
        initial_state=question_states_tiled
        if options.text_feature == model_pb2.QUESTION_AND_ANSWER else None)

    answer_choices_feature = tf.reshape(answer_choices_feature_reshaped,
                                        [batch_size, NUM_CHOICES, -1])

    output = tf.keras.layers.Dense(1, activation=None)(answer_choices_feature)
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
    return tf.compat.v1.trainable_variables()
