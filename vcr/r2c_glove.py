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
from modeling.layers import token_to_id
import tf_slim as slim

FIELD_ANSWER_PREDICTION = 'answer_prediction'


def _load_embeddings(filename):
  """Loads embedding vectors from GloVe file.

  Args:
    filename: Path to the GloVe word embedding file.

  Returns:
    A word embedding dict keyed by word.
  """
  embeddings_index = {}
  with open(filename, 'r', encoding='utf8') as f:
    for i, line in enumerate(f):
      word, coefs = line.split(maxsplit=1)
      coefs = np.fromstring(coefs, dtype=np.float32, sep=' ')
      embeddings_index[word] = coefs
      if (i + 1) % 10000 == 0:
        logging.info('Load embedding %i.', i + 1)
  return embeddings_index


def _load_vocabulary(filename):
  """Loads vocabulary file.

  Args:
    filename: Path to the vocabulary file, each line is a word.

  Returns:
    A list of strings.
  """
  with open(filename, 'r', encoding='utf8') as f:
    return [x.strip('\n') for x in f]


def _create_embedding_matrix(glove_file, vocab_file, init_width=0.03):
  if glove_file:
    embeddings_index = _load_embeddings(glove_file)
    embedding_dims = embeddings_index['the'].shape[-1]
  else:
    embeddings_index = dict()
    embedding_dims = 300
  vocab = _load_vocabulary(vocab_file)

  embedding_matrix = np.random.uniform(-init_width, init_width,
                                       (len(vocab), embedding_dims))
  for i, word in enumerate(vocab):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  return embedding_matrix.astype(np.float32)


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
                                 embedding_dims=300,
                                 scope='object_embedding',
                                 max_norm=None):
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
  with tf.variable_scope(scope):
    object_embedding = tf.get_variable('weights',
                                       shape=[vocab_size + 1, embedding_dims],
                                       trainable=True)
  return tf.nn.embedding_lookup(object_embedding, label_ids, max_norm=max_norm)


def _project_object_features(object_features,
                             object_embeddings,
                             output_dims,
                             dropout_keep_prob=1.0,
                             is_training=False):
  """Projects object features to `visual_feature_dims` dimensions.

  Args:
    object_features: A [batch, max_num_objects, feature_dims] float tensor.
    object_embeddings: A [batch, max_num_objects, embedding_dims] float tensor.

  Returns:
    A [batch, max_num_objects, output_dims] float tensor.
  """
  object_features = tf.concat([object_features, object_embeddings], -1)

  with tf.variable_scope('object_projection'):
    object_features = tf.contrib.layers.fully_connected(
        object_features, num_outputs=output_dims, activation_fn=tf.nn.relu)
    object_features = tf.contrib.layers.dropout(object_features,
                                                keep_prob=dropout_keep_prob,
                                                is_training=is_training)
  return object_features


def _reshape_answer_related_tensors(question_embs, question_tags, question_len,
                                    answer_embs, answer_tags, answer_len):
  """Reshapes answer related tensors to [batch * NUM_CHOICES, max_seq_len, ...].

  Args:
    question_embs: A [batch, max_question_len, embedding_dims] tensor.
    question_tags: A [batch, max_question_len] int tensor.
    question_len: A [batch] int tensor.
    answer_embs: A [batch, NUM_CHOICES, max_answer_len, embedding_dims] tensor.
    answer_tags: A [batch, NUM_CHOICES, max_answer_len] int tensor.
    answer_len: A [batch, NUM_CHOICES] int tensor.

  Returns:
    The same tensors with size reshaped to [batch * NUM_CHOICES, ...].
  """
  batch_size, embedding_dims = question_embs.shape[0], question_embs.shape[-1]

  # Tile, expand a dimension.
  tile_fn = lambda x: tf.gather(tf.expand_dims(x, 1), [0] * NUM_CHOICES, axis=1)
  (question_embs, question_tags, question_len) = (tile_fn(question_embs),
                                                  tile_fn(question_tags),
                                                  tile_fn(question_len))

  # Reshape question tensors.
  question_embs = tf.reshape(question_embs,
                             [batch_size * NUM_CHOICES, -1, embedding_dims])
  question_tags = tf.reshape(question_tags, [batch_size * NUM_CHOICES, -1])
  question_len = tf.reshape(question_len, [batch_size * NUM_CHOICES])

  # Reshape answer tensors.
  answer_embs = tf.reshape(answer_embs,
                           [batch_size * NUM_CHOICES, -1, embedding_dims])
  answer_tags = tf.reshape(answer_tags, [batch_size * NUM_CHOICES, -1])
  answer_len = tf.reshape(answer_len, [batch_size * NUM_CHOICES])

  return (question_embs, question_tags, question_len, answer_embs, answer_tags,
          answer_len)


def _ground_tag_using_object_features(object_features, tags):
  """Grounds tag sequence using object features.

  Args:
    object_features: A [batch, max_num_objects, feature_dims] float tensor.
    tags: A [batch * NUM_CHOICES, max_seq_len] int tensor.
  """
  # Add ZEROs to the end.
  batch_size, _, object_feature_dims = object_features.shape
  object_features_padded = tf.concat(
      [object_features,
       tf.zeros([batch_size, 1, object_feature_dims])], 1)

  # Tile object features.
  object_features_padded_and_tiled = tf.reshape(
      tf.gather(tf.expand_dims(object_features_padded, 1), [0] * NUM_CHOICES,
                axis=1), [batch_size * NUM_CHOICES, -1, object_feature_dims])

  # Gather tag embeddings.
  feature_indices = tf.stack([
      tf.tile(tf.expand_dims(tf.range(batch_size * NUM_CHOICES), axis=1),
              [1, tf.shape(tags)[1]]), tags
  ], -1)
  grounded_object_features = tf.gather_nd(object_features_padded_and_tiled,
                                          feature_indices)
  return grounded_object_features


class VCRR2CGloVe(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VCRR2CGloVe, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VCRR2CGloVe):
      raise ValueError('Options has to be an VCRR2CGloVe proto.')

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

  def _recognition_to_cognition(self, question_inp_features, question_len,
                                answer_inp_features, answer_len,
                                object_features, num_objects, predictions):
    """Creates the `RecognitionToCognition` network.

    Args:
      question_inp_features: Input question features, a [batch*NUM_CHOICES,
        max_question_len, feature_dims] float tensor.
      question_len: Question length, a [batch*NUM_CHOICES] int tensor.
      answer_inp_features: Input answer features, a [batch*NUM_CHOICES,
        max_answer_len , feature_dims] float tensor.
      answer_len: Answer length, a [batch*NUM_CHOICES] int tensor.
      object_features: Object features, a [batch, max_num_objects, object_dims]
        float tensor.
      num_objects: A [batch] int tensor.

    Returns:
      final_features: A [batch, output_dims] float tensor.
      answer_seq_features: Contextualized answer features, a [batch*NUM_CHOICES,
        max_answer_len , feature_dims] float tensor.
    """
    is_training = self._is_training
    options = self._model_proto

    (question_max_len, answer_max_len) = (tf.shape(question_inp_features)[1],
                                          tf.shape(answer_inp_features)[1])
    batch_size = object_features.shape[0]
    max_num_objects = tf.shape(object_features)[1]

    # Encode the sequence using BiLSTM model.
    with tf.variable_scope('grounding_encoder'):
      question_seq_features, _ = rnn.RNN(question_inp_features,
                                         question_len,
                                         options.rnn_config,
                                         is_training=is_training)

    with tf.variable_scope('grounding_encoder', reuse=True):
      answer_seq_features, _ = rnn.RNN(answer_inp_features,
                                       answer_len,
                                       options.rnn_config,
                                       is_training=is_training)

    # Get the question features attended by the answers.
    #   qa_mask: [batch*NUM_CHOICES, question_max_len, 1].
    #   qa_similarity: [batch*NUM_CHOICES, question_max_len, answer_max_len].
    #   qa_attention_weights: [batch*NUM_CHOICES, question_max_len, answer_max_len].
    #   attended_question: [batch*NUM_CHOICES, answer_max_len, feature_dims].
    qa_mask = tf.expand_dims(
        tf.sequence_mask(question_len, question_max_len, dtype=tf.float32), 2)
    with tf.variable_scope('qa_bilinear'):
      qa_similarity = attention_ops.bilinear(question_seq_features,
                                             answer_seq_features)
    qa_attention_weights = masked_ops.masked_softmax(data=qa_similarity,
                                                     mask=qa_mask,
                                                     dim=1)
    attended_question = tf.einsum('bqa,bqd->bad', qa_attention_weights,
                                  question_seq_features)

    # Attention over the objects.
    #   oa_mask: [batch, max_num_object, 1].
    #   oa_similarity: [batch*NUM_CHOICES, max_num_object, answer_max_len]
    #   oa_attention_weights: [batch*NUM_CHOICES, max_num_object, answer_max_len].
    #   attended_objects: [batch*NUM_CHOICES, answer_max_len, object_dims].

    tile_fn = lambda x: tf.gather(tf.expand_dims(x, 1), [0] * NUM_CHOICES,
                                  axis=1)
    object_features = tf.reshape(
        tile_fn(object_features),
        [batch_size * NUM_CHOICES, -1, object_features.shape[-1]])
    num_objects = tf.reshape(tile_fn(num_objects), [-1])

    oa_mask = tf.expand_dims(
        tf.sequence_mask(num_objects, max_num_objects, dtype=tf.float32), 2)
    with tf.variable_scope('oa_bilinear'):
      oa_similarity = attention_ops.bilinear(object_features,
                                             answer_seq_features)
    oa_attention_weights = masked_ops.masked_softmax(data=oa_similarity,
                                                     mask=oa_mask,
                                                     dim=1)
    attended_objects = tf.einsum('boa,bod->bad', oa_attention_weights,
                                 object_features)

    # Reasoning module.
    reasoning_inp_features = tf.concat(
        [answer_seq_features, attended_question, attended_objects], -1)

    with tf.variable_scope('reasoning'):
      reasoning_seq_features, _ = rnn.RNN(reasoning_inp_features,
                                          answer_len,
                                          options.rnn_config,
                                          is_training=is_training)

    # Pool features from the sequence.
    pooling_fn = (masked_ops.masked_max_nd
                  if options.use_max_pooling else masked_ops.masked_avg_nd)

    final_seq_features = tf.concat([
        reasoning_seq_features, answer_seq_features, attended_question,
        attended_objects
    ], -1)
    final_features = pooling_fn(data=final_seq_features,
                                mask=tf.sequence_mask(answer_len,
                                                      answer_max_len,
                                                      dtype=tf.float32),
                                dim=1)

    # Export summaries.
    tf.compat.v1.summary.histogram('attention/qa_similarity', qa_similarity)
    tf.compat.v1.summary.histogram('attention/oa_similarity', oa_similarity)
    predictions.update({
        'r2c/num_objects': num_objects,
        'r2c/oa_mask': oa_mask,
        'r2c/qa_mask': qa_mask,
        'r2c/oa_attention_weights': oa_attention_weights,
        'r2c/qa_attention_weights': qa_attention_weights,
    })
    return (tf.squeeze(final_features, 1), answer_seq_features)

  def _predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto
    token_to_id_layer = token_to_id.TokenToIdLayer(
        options.embedding_vocab_file, options.embedding_unk_token_id)

    predictions = {}

    # Extract text annotations.
    (question, question_len, answer,
     answer_len) = (inputs[InputFields.question],
                    inputs[InputFields.question_len],
                    inputs[InputFields.answer_choices],
                    inputs[InputFields.answer_choices_len])
    batch_size = question.shape[0]

    # Convert word to embedding vectors.
    (question_token_ids, answer_token_ids) = (token_to_id_layer(question),
                                              token_to_id_layer(answer))
    glove_embedding_array = _create_embedding_matrix(
        options.embedding_glove_file, options.embedding_vocab_file)
    embedding = tf.get_variable('word/embedding',
                                initializer=glove_embedding_array,
                                trainable=True)

    question_embs = tf.nn.embedding_lookup(embedding,
                                           question_token_ids,
                                           max_norm=None)
    answer_embs = tf.nn.embedding_lookup(embedding,
                                         answer_token_ids,
                                         max_norm=None)

    # Trim lengths of the object arrays to `max_num_objects`.
    (num_objects, object_bboxes, object_labels, object_scores, object_features,
     max_num_objects) = _trim_to_max_num_objects(
         inputs[InputFields.num_objects],
         inputs[InputFields.object_bboxes],
         inputs[InputFields.object_labels],
         inputs[InputFields.object_scores],
         inputs[InputFields.object_features],
         max_num_objects=options.max_num_objects)

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
    (question_embs, question_tags, question_len, answer_embs, answer_tags,
     answer_len) = _reshape_answer_related_tensors(question_embs, question_tags,
                                                   question_len, answer_embs,
                                                   answer_tags, answer_len)

    # Ground both the question and the answer choices.
    question_object_features = _ground_tag_using_object_features(
        object_features, question_tags)
    answer_object_features = _ground_tag_using_object_features(
        object_features, answer_tags)
    question_rnn_inputs = tf.concat([question_embs, question_object_features],
                                    -1)
    answer_rnn_inputs = tf.concat([answer_embs, answer_object_features], -1)

    # Build the recognition to cognition model.
    final_features, answer_seq_features = self._recognition_to_cognition(
        question_rnn_inputs, question_len, answer_rnn_inputs, answer_len,
        object_features, num_objects, predictions)

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

    predictions.update({
        FIELD_ANSWER_PREDICTION: output,
        'image_id': inputs[InputFields.img_id]
    })
    return predictions

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
